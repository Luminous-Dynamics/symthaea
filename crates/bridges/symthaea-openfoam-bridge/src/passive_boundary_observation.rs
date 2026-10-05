// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative observation of an OpenFOAM constant/polyMesh/boundary artifact.
//!
//! This module proves only conservative provenance facts about supplied OpenFOAM
//! input artifacts. It does not claim that a live solver loaded the files, that
//! solver geometry matches the candidate mesh, or that any physics ran.

use blake3::Hasher;
use symthaea_passive_solver_binding::{SolverBoundaryEntityObservation, SolverBindingError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenFoamBoundaryPatchRecord {
    pub patch_name: String,
    pub patch_type: String,
    pub n_faces: u64,
    pub start_face: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenFoamBoundaryObservationError {
    InvalidUtf8,
    UnterminatedBlockComment,
    UnexpectedCharacter(u8),
    UnexpectedEndOfInput,
    DuplicatePatch(String),
    DuplicatePatchField { patch: String, field: &'static str },
    PatchNotFound(String),
    MissingPatchField { patch: String, field: &'static str },
    InvalidNumericField { patch: String, field: &'static str },
    InvalidPatchType(String),
    ArithmeticOverflow,
    MissingPatchList,
    DuplicatePatchList,
    PatchCountMismatch { declared: u64, observed: u64 },
    MissingFaceList,
    DuplicateFaceList,
    InvalidFaceListEntry,
    FacesCountMismatch { declared: u64, observed: u64 },
    InvalidFaceRecord { face_index: u64 },
    DegenerateFace { face_index: u64 },
    FaceNormalMismatch { face_index: u64 },
    NonManifoldPatchEdge { face_index: u64, point_a: u64, point_b: u64 },
    BoundaryFaceRangeOutOfBounds {
        start_face: u64,
        n_faces: u64,
        face_count: u64,
    },
    MissingPointList,
    DuplicatePointList,
    InvalidPointRecord { point_index: u64 },
    PointsCountMismatch { declared: u64, observed: u64 },
    PointIndexOutOfBounds { face_index: u64, point_index: u64 },
    MissingNeighbourList,
    DuplicateNeighbourList,
    NeighbourCountMismatch { declared: u64, observed: u64 },
    InvalidNeighbourListEntry,
    MissingOwnerList,
    DuplicateOwnerList,
    OwnerCountMismatch { declared: u64, observed: u64 },
    InvalidOwnerListEntry,
    InternalFaceSelfLoop { face_index: u64, cell: u64 },
    NonContiguousBoundaryPatchRange {
        patch: String,
        expected_start: u64,
        actual_start: u64,
    },
    BoundaryPatchRangeOverlap { patch: String },
    BoundaryPatchRangeExceedsFaces {
        patch: String,
        end_face: u64,
        face_count: u64,
    },
    InvalidPointScale,
    InvalidArtifactCombination,
    PatchGeometryMismatch,
    UnexpectedPatchListEntry,
    InvalidTolerance,
    InvalidSolverBinding(SolverBindingError),
}

impl From<SolverBindingError> for OpenFoamBoundaryObservationError {
    fn from(error: SolverBindingError) -> Self {
        Self::InvalidSolverBinding(error)
    }
}

impl OpenFoamBoundaryPatchRecord {
    pub fn canonical_identity_bytes(&self) -> Vec<u8> {
        fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value);
        }

        let mut out = Vec::new();
        out.extend_from_slice(b"openfoam-polyMesh-boundary-patch:v1");
        put_bytes(&mut out, self.patch_name.as_bytes());
        put_bytes(&mut out, self.patch_type.as_bytes());
        out.extend_from_slice(&self.start_face.to_le_bytes());
        out.extend_from_slice(&self.n_faces.to_le_bytes());
        out
    }

    pub fn observe(
        &self,
        source_bytes: &[u8],
    ) -> Result<SolverBoundaryEntityObservation, OpenFoamBoundaryObservationError> {
        if self.patch_name.trim().is_empty() {
            return Err(OpenFoamBoundaryObservationError::PatchNotFound(self.patch_name.clone()));
        }
        if self.patch_type.trim().is_empty() {
            return Err(OpenFoamBoundaryObservationError::InvalidPatchType(self.patch_name.clone()));
        }

        let mut hasher = Hasher::new();
        hasher.update(b"openfoam-polyMesh-boundary-source:v1");
        hasher.update(&(source_bytes.len() as u64).to_le_bytes());
        hasher.update(source_bytes);
        let source_digest = *hasher.finalize().as_bytes();

        SolverBoundaryEntityObservation::new(
            "openfoam-polyMesh-boundary-patch:v1",
            self.canonical_identity_bytes(),
            source_digest,
        )
        .map_err(Into::into)
    }
}

pub fn observe_openfoam_boundary_patch(
    source_bytes: &[u8],
    patch_name: &str,
) -> Result<(OpenFoamBoundaryPatchRecord, OpenFoamBoundaryEntityObservation), OpenFoamBoundaryObservationError> {
    let text = std::str::from_utf8(source_bytes)
        .map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?;
    let stripped = strip_comments(text)?;
    let tokens = tokenize(&stripped)?;
    let records = parse_boundary_patch_list(&tokens)?;

    match records.iter().find(|record| record.patch_name == patch_name) {
        Some(record) => Ok((record.clone(), record.observe(source_bytes)?)),
        None => Err(OpenFoamBoundaryObservationError::PatchNotFound(
            patch_name.to_string(),
        )),
    }
}

type OpenFoamBoundaryEntityObservation = SolverBoundaryEntityObservation;

fn observe_openfoam_boundary_patch_and_faces(
    boundary_source_bytes: &[u8],
    faces_source_bytes: &[u8],
    patch_name: &str,
) -> Result<
    (OpenFoamBoundaryPatchRecord, OpenFoamBoundaryEntityObservation),
    OpenFoamBoundaryObservationError,
> {
    let (record, _) =
        observe_openfoam_boundary_patch(boundary_source_bytes, patch_name)?;
    let faces = parse_face_list(faces_source_bytes)?;
    let face_count = faces.len() as u64;

    let end_face = record
        .start_face
        .checked_add(record.n_faces)
        .ok_or(OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    if end_face > face_count {
        return Err(
            OpenFoamBoundaryObservationError::BoundaryFaceRangeOutOfBounds {
                start_face: record.start_face,
                n_faces: record.n_faces,
                face_count,
            },
        );
    }

    let mut source_hasher = Hasher::new();
    source_hasher.update(b"openfoam-polyMesh-boundary-and-faces-source:v1");
    source_hasher.update(&(boundary_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(boundary_source_bytes);
    source_hasher.update(&(faces_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(faces_source_bytes);
    let source_digest = *source_hasher.finalize().as_bytes();

    let mut identity = Vec::new();
    identity.extend_from_slice(b"openfoam-polyMesh-boundary-patch-and-face-list:v1");
    let boundary_identity = record.canonical_identity_bytes();
    identity.extend_from_slice(&(boundary_identity.len() as u64).to_le_bytes());
    identity.extend_from_slice(&boundary_identity);
    identity.extend_from_slice(&face_count.to_le_bytes());

    let observation = SolverBoundaryEntityObservation::new(
        "openfoam-polyMesh-boundary-patch-and-face-list:v1",
        identity,
        source_digest,
    )
    .map_err(Into::into)?;

    Ok((record, observation))
}

fn parse_face_list(
    source_bytes: &[u8],
) -> Result<Vec<Vec<u64>>, OpenFoamBoundaryObservationError> {
    let text = std::str::from_utf8(source_bytes)
        .map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?;
    let stripped = strip_comments(text)?;
    let tokens = tokenize(&stripped)?;

    let (count, mut index) = locate_top_level_list(
        &tokens,
        OpenFoamBoundaryObservationError::MissingFaceList,
        OpenFoamBoundaryObservationError::DuplicateFaceList,
        OpenFoamBoundaryObservationError::InvalidFaceListEntry,
    )?;

    let mut faces = Vec::new();
    while index < tokens.len() && !matches!(tokens[index], Token::RParen) {
        let face_index = faces.len() as u64;
        let Token::Number(vertex_count) = tokens.get(index).ok_or(
            OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index },
        )? else {
            return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index });
        };
        let vertex_count = vertex_count.parse::<usize>().map_err(|_| {
            OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index }
        })?;
        if vertex_count < 3 {
            return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index });
        }
        index += 1;
        if !matches!(tokens.get(index), Some(Token::LParen)) {
            return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index });
        }
        index += 1;

        let mut face = Vec::with_capacity(vertex_count);
        for _ in 0..vertex_count {
            let Token::Number(point) = tokens.get(index).ok_or(
                OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index },
            )? else {
                return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index });
            };
            let point = point.parse::<u64>().map_err(|_| {
                OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index }
            })?;
            face.push(point);
            index += 1;
        }

        if !matches!(tokens.get(index), Some(Token::RParen)) {
            return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index });
        }
        index += 1;
        faces.push(face);
    }

    if !matches!(tokens.get(index), Some(Token::RParen)) {
        return Err(OpenFoamBoundaryObservationError::InvalidFaceListEntry);
    }
    index += 1;

    if index != tokens.len() {
        return Err(OpenFoamBoundaryObservationError::InvalidFaceListEntry);
    }

    let observed = faces.len() as u64;
    if observed != count {
        return Err(OpenFoamBoundaryObservationError::FacesCountMismatch {
            declared: count,
            observed,
        });
    }
    Ok(faces)
}

fn parse_neighbour_list(
    source_bytes: &[u8],
) -> Result<Vec<u64>, OpenFoamBoundaryObservationError> {
    let text = std::str::from_utf8(source_bytes)
        .map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?;
    let stripped = strip_comments(text)?;
    let tokens = tokenize(&stripped)?;

    let (count, mut index) = locate_top_level_list(
        &tokens,
        OpenFoamBoundaryObservationError::MissingNeighbourList,
        OpenFoamBoundaryObservationError::DuplicateNeighbourList,
        OpenFoamBoundaryObservationError::InvalidNeighbourListEntry,
    )?;

    let mut neighbours = Vec::new();
    while index < tokens.len() && !matches!(tokens[index], Token::RParen) {
        let Token::Number(value) = tokens.get(index).ok_or(
            OpenFoamBoundaryObservationError::InvalidNeighbourListEntry,
        )? else {
            return Err(OpenFoamBoundaryObservationError::InvalidNeighbourListEntry);
        };
        let value = value.parse::<u64>().map_err(|_| {
            OpenFoamBoundaryObservationError::InvalidNeighbourListEntry
        })?;
        neighbours.push(value);
        index += 1;
    }

    if !matches!(tokens.get(index), Some(Token::RParen)) || index + 1 != tokens.len() {
        return Err(OpenFoamBoundaryObservationError::InvalidNeighbourListEntry);
    }

    let observed = neighbours.len() as u64;
    if observed != count {
        return Err(OpenFoamBoundaryObservationError::NeighbourCountMismatch {
            declared: count,
            observed,
        });
    }
    Ok(neighbours)
}

fn parse_neighbour_list_count(
    source_bytes: &[u8],
) -> Result<u64, OpenFoamBoundaryObservationError> {
    Ok(parse_neighbour_list(source_bytes)?.len() as u64)
}

fn parse_owner_list(
    source_bytes: &[u8],
) -> Result<Vec<u64>, OpenFoamBoundaryObservationError> {
    let text = std::str::from_utf8(source_bytes)
        .map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?;
    let stripped = strip_comments(text)?;
    let tokens = tokenize(&stripped)?;

    let (count, mut index) = locate_top_level_list(
        &tokens,
        OpenFoamBoundaryObservationError::MissingOwnerList,
        OpenFoamBoundaryObservationError::DuplicateOwnerList,
        OpenFoamBoundaryObservationError::InvalidOwnerListEntry,
    )?;

    let mut owners = Vec::new();
    while index < tokens.len() && !matches!(tokens[index], Token::RParen) {
        let Token::Number(value) = tokens.get(index).ok_or(
            OpenFoamBoundaryObservationError::InvalidOwnerListEntry,
        ) else {
            return Err(OpenFoamBoundaryObservationError::InvalidOwnerListEntry);
        };
        let value = value.parse::<u64>().map_err(|_| {
            OpenFoamBoundaryObservationError::InvalidOwnerListEntry
        })?;
        owners.push(value);
        index += 1;
    }

    if !matches!(tokens.get(index), Some(Token::RParen)) || index + 1 != tokens.len() {
        return Err(OpenFoamBoundaryObservationError::InvalidOwnerListEntry);
    }

    let observed = owners.len() as u64;
    if observed != count {
        return Err(OpenFoamBoundaryObservationError::OwnerCountMismatch {
            declared: count,
            observed,
        });
    }
    Ok(owners)
}

fn parse_owner_list_count(
    source_bytes: &[u8],
) -> Result<u64, OpenFoamBoundaryObservationError> {
    Ok(parse_owner_list(source_bytes)?.len() as u64)
}

fn validate_internal_face_cells(
    owners: &[u64],
    neighbours: &[u64],
) -> Result<(), OpenFoamBoundaryObservationError> {
    if neighbours.len() > owners.len() {
        return Err(
            OpenFoamBoundaryObservationError::BoundaryPatchRangeExceedsFaces {
                patch: "<internal-faces>".to_string(),
                end_face: neighbours.len() as u64,
                face_count: owners.len() as u64,
            },
        );
    }

    for (face_index, (&owner, &neighbour)) in owners.iter().zip(neighbours).enumerate() {
        if owner == neighbour {
            return Err(OpenFoamBoundaryObservationError::InternalFaceSelfLoop {
                face_index: face_index as u64,
                cell: owner,
            });
        }
    }
    Ok(())
}

fn validate_boundary_patch_partition(
    records: &[OpenFoamBoundaryPatchRecord],
    internal_face_count: u64,
    face_count: u64,
) -> Result<(), OpenFoamBoundaryObservationError> {
    let mut expected_start = internal_face_count;
    for record in records {
        let end_face = record
            .start_face
            .checked_add(record.n_faces)
            .ok_or(OpenFoamBoundaryObservationError::ArithmeticOverflow)?;

        if record.start_face < expected_start {
            return Err(
                OpenFoamBoundaryObservationError::BoundaryPatchRangeOverlap {
                    patch: record.patch_name.clone(),
                },
            );
        }
        if record.start_face != expected_start {
            return Err(
                OpenFoamBoundaryObservationError::NonContiguousBoundaryPatchRange {
                    patch: record.patch_name.clone(),
                    expected_start,
                    actual_start: record.start_face,
                },
            );
        }
        if end_face > face_count {
            return Err(
                OpenFoamBoundaryObservationError::BoundaryPatchRangeExceedsFaces {
                    patch: record.patch_name.clone(),
                    end_face,
                    face_count,
                },
            );
        }
        expected_start = end_face;
    }

    if expected_start != face_count {
        return Err(
            OpenFoamBoundaryObservationError::NonContiguousBoundaryPatchRange {
                patch: "<boundary-end>".to_string(),
                expected_start,
                actual_start: face_count,
            },
        );
    }
    Ok(())
}

fn observe_openfoam_patch_geometry(
    boundary_source_bytes: &[u8],
    faces_source_bytes: &[u8],
    points_source_bytes: &[u8],
    patch_name: &str,
    interface: &symthaea_passive_void_compiler::PortInterface,
    candidate: &symthaea_fabrication_kernel::mesh::TriangleMesh,
    tolerance_mm: f64,
    point_scale_mm_per_unit: f64,
) -> Result<
    (OpenFoamBoundaryPatchRecord, OpenFoamBoundaryEntityObservation),
    OpenFoamBoundaryObservationError,
> {
    if !point_scale_mm_per_unit.is_finite() || point_scale_mm_per_unit <= 0.0 {
        return Err(OpenFoamBoundaryObservationError::InvalidPointScale);
    }

    let (record, _) =
        observe_openfoam_boundary_patch(boundary_source_bytes, patch_name)?;
    let faces = parse_face_list(faces_source_bytes)?;
    let points = parse_points_list(points_source_bytes)?;

    let start = usize::try_from(record.start_face)
        .map_err(|_| OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    let count = usize::try_from(record.n_faces)
        .map_err(|_| OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    let end = start
        .checked_add(count)
        .ok_or(OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    if end > faces.len() {
        return Err(OpenFoamBoundaryObservationError::BoundaryFaceRangeOutOfBounds {
            start_face: record.start_face,
            n_faces: record.n_faces,
            face_count: faces.len() as u64,
        });
    }

    let mut edge_occurrences =
        std::collections::BTreeMap::<symthaea_passive_solver_binding::BoundaryEdgeKey, u32>::new();
    let mut edge_origins =
        std::collections::BTreeMap::<symthaea_passive_solver_binding::BoundaryEdgeKey, (u64, u64)>::new();

    for (offset, face) in faces[start..end].iter().enumerate() {
        if face.len() < 3 {
            return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord {
                face_index: record.start_face + offset as u64,
            });
        }

        let mut seen_points = std::collections::BTreeSet::new();
        for &point_index in face {
            if !seen_points.insert(point_index) {
                return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord {
                    face_index: record.start_face + offset as u64,
                });
            }
        }

        let mut normal = [0.0f64; 3];
        for index in 0..face.len() {
            let a = points
                .get(usize::try_from(face[index]).map_err(|_| {
                    OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                        face_index: record.start_face + offset as u64,
                        point_index: face[index],
                    }
                })?)
                .ok_or(OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                    face_index: record.start_face + offset as u64,
                    point_index: face[index],
                })?;
            let b = points
                .get(usize::try_from(face[(index + 1) % face.len()]).map_err(|_| {
                    OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                        face_index: record.start_face + offset as u64,
                        point_index: face[(index + 1) % face.len()],
                    }
                })?)
                .ok_or(OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                    face_index: record.start_face + offset as u64,
                    point_index: face[(index + 1) % face.len()],
                })?;
            normal[0] += (a[1] - b[1]) * (a[2] + b[2]);
            normal[1] += (a[2] - b[2]) * (a[0] + b[0]);
            normal[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        let normal_magnitude = (normal[0] * normal[0]
            + normal[1] * normal[1]
            + normal[2] * normal[2]).sqrt();
        if !normal_magnitude.is_finite() || normal_magnitude <= 1.0e-12 {
            return Err(OpenFoamBoundaryObservationError::DegenerateFace {
                face_index: record.start_face + offset as u64,
            });
        }
        let normal_alignment =
            (normal[0] / normal_magnitude) * interface.outward_normal_unit[0] as f64
                + (normal[1] / normal_magnitude) * interface.outward_normal_unit[1] as f64
                + (normal[2] / normal_magnitude) * interface.outward_normal_unit[2] as f64;
        if !normal_alignment.is_finite() || normal_alignment <= 0.0 {
            return Err(OpenFoamBoundaryObservationError::FaceNormalMismatch {
                face_index: record.start_face + offset as u64,
            });
        }
        for edge_index in 0..face.len() {
            let a_index = face[edge_index];
            let b_index = face[(edge_index + 1) % face.len()];
            let a = points.get(usize::try_from(a_index).map_err(|_| {
                OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                    face_index: record.start_face + offset as u64,
                    point_index: a_index,
                }
            })?).ok_or(OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                face_index: record.start_face + offset as u64,
                point_index: a_index,
            })?;
            let b = points.get(usize::try_from(b_index).map_err(|_| {
                OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                    face_index: record.start_face + offset as u64,
                    point_index: b_index,
                }
            })?).ok_or(OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                face_index: record.start_face + offset as u64,
                point_index: b_index,
            })?;

            let a_mm = scaled_point_to_f32(*a, point_scale_mm_per_unit)?;
            let b_mm = scaled_point_to_f32(*b, point_scale_mm_per_unit)?;
            let edge = symthaea_passive_solver_binding::BoundaryEdgeKey::new(a_mm, b_mm)
                .map_err(|_| OpenFoamBoundaryObservationError::PatchGeometryMismatch)?;
            let topology_edge = if a_index <= b_index {
                (a_index, b_index)
            } else {
                (b_index, a_index)
            };
            if let Some(existing) = edge_origins.get(&edge) {
                if *existing != topology_edge {
                    return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
                }
            } else {
                edge_origins.insert(edge, topology_edge);
            }
            let entry = edge_occurrences.entry(edge).or_insert(0);
            *entry = entry.checked_add(1).ok_or(
                OpenFoamBoundaryObservationError::ArithmeticOverflow,
            )?;
            if *entry > 2 {
                return Err(OpenFoamBoundaryObservationError::NonManifoldPatchEdge {
                    face_index: record.start_face + offset as u64,
                    point_a: a_index,
                    point_b: b_index,
                });
            }
        }
    }

    let solver_boundary_edges: std::collections::BTreeSet<_> = edge_occurrences
        .into_iter()
        .filter_map(|(edge, occurrences)| (occurrences == 1).then_some(edge))
        .collect();
    if solver_boundary_edges.is_empty() {
        return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
    }

    let candidate_selection = symthaea_passive_solver_binding::select_boundary_patch(
        interface,
        candidate,
        tolerance_mm,
    ).map_err(|_| OpenFoamBoundaryObservationError::PatchGeometryMismatch)?;
    let candidate_edges: std::collections::BTreeSet<_> =
        candidate_selection.edges().iter().copied().collect();
    if solver_boundary_edges != candidate_edges {
        return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
    }

    let mut source_hasher = Hasher::new();
    source_hasher.update(b"openfoam-polyMesh-patch-geometry-source:v1");
    source_hasher.update(&(boundary_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(boundary_source_bytes);
    source_hasher.update(&(faces_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(faces_source_bytes);
    source_hasher.update(&(points_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(points_source_bytes);
    let source_digest = *source_hasher.finalize().as_bytes();

    let mut identity = Vec::new();
    identity.extend_from_slice(b"openfoam-polyMesh-patch-geometry:v1");
    let boundary_identity = record.canonical_identity_bytes();
    identity.extend_from_slice(&(boundary_identity.len() as u64).to_le_bytes());
    identity.extend_from_slice(&boundary_identity);
    identity.extend_from_slice(&point_scale_mm_per_unit.to_le_bytes());
    identity.extend_from_slice(&tolerance_mm.to_le_bytes());
    identity.extend_from_slice(&(solver_boundary_edges.len() as u64).to_le_bytes());
    for edge in &solver_boundary_edges {
        identity.extend_from_slice(&edge.a[0].to_le_bytes());
        identity.extend_from_slice(&edge.a[1].to_le_bytes());
        identity.extend_from_slice(&edge.a[2].to_le_bytes());
        identity.extend_from_slice(&edge.b[0].to_le_bytes());
        identity.extend_from_slice(&edge.b[1].to_le_bytes());
        identity.extend_from_slice(&edge.b[2].to_le_bytes());
    }

    let observation = SolverBoundaryEntityObservation::new(
        "openfoam-polyMesh-patch-geometry:v1",
        identity,
        source_digest,
    ).map_err(Into::into)?;

    Ok((record, observation))
}

fn observe_openfoam_patch_geometry_with_owner_and_neighbour(
    boundary_source_bytes: &[u8],
    faces_source_bytes: &[u8],
    points_source_bytes: &[u8],
    neighbour_source_bytes: &[u8],
    owner_source_bytes: &[u8],
    patch_name: &str,
    interface: &symthaea_passive_void_compiler::PortInterface,
    candidate: &symthaea_fabrication_kernel::mesh::TriangleMesh,
    tolerance_mm: f64,
    point_scale_mm_per_unit: f64,
) -> Result<
    (OpenFoamBoundaryPatchRecord, OpenFoamBoundaryEntityObservation),
    OpenFoamBoundaryObservationError,
> {
    let faces = parse_face_list(faces_source_bytes)?;
    let owners = parse_owner_list(owner_source_bytes)?;
    if owners.len() != faces.len() {
        return Err(OpenFoamBoundaryObservationError::OwnerCountMismatch {
            declared: owners.len() as u64,
            observed: faces.len() as u64,
        });
    }
    let neighbours = parse_neighbour_list(neighbour_source_bytes)?;
    validate_internal_face_cells(&owners, &neighbours)?;

    let (record, observation) = observe_openfoam_patch_geometry_with_neighbour(
        boundary_source_bytes,
        faces_source_bytes,
        points_source_bytes,
        neighbour_source_bytes,
        patch_name,
        interface,
        candidate,
        tolerance_mm,
        point_scale_mm_per_unit,
    )?;

    let mut source_hasher = Hasher::new();
    source_hasher.update(b"openfoam-polyMesh-patch-geometry-with-owner-neighbour-source:v1");
    source_hasher.update(&(owner_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(owner_source_bytes);
    source_hasher.update(&observation.source_digest);

    let mut identity = Vec::new();
    identity.extend_from_slice(b"openfoam-polyMesh-patch-geometry-with-owner-neighbour:v1");
    identity.extend_from_slice(&observation.digest());

    let observation = SolverBoundaryEntityObservation::new(
        "openfoam-polyMesh-patch-geometry-with-owner-neighbour:v1",
        identity,
        *source_hasher.finalize().as_bytes(),
    )
    .map_err(Into::into)?;

    Ok((record, observation))
}

fn observe_openfoam_patch_geometry_with_neighbour(
    boundary_source_bytes: &[u8],
    faces_source_bytes: &[u8],
    points_source_bytes: &[u8],
    neighbour_source_bytes: &[u8],
    patch_name: &str,
    interface: &symthaea_passive_void_compiler::PortInterface,
    candidate: &symthaea_fabrication_kernel::mesh::TriangleMesh,
    tolerance_mm: f64,
    point_scale_mm_per_unit: f64,
) -> Result<
    (OpenFoamBoundaryPatchRecord, OpenFoamBoundaryEntityObservation),
    OpenFoamBoundaryObservationError,
> {
    if !point_scale_mm_per_unit.is_finite() || point_scale_mm_per_unit <= 0.0 {
        return Err(OpenFoamBoundaryObservationError::InvalidPointScale);
    }

    let (record, _) =
        observe_openfoam_boundary_patch(boundary_source_bytes, patch_name)?;
    let boundary_text = std::str::from_utf8(boundary_source_bytes)
        .map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?;
    let boundary_tokens = tokenize(&strip_comments(boundary_text)?)?;
    let boundary_records = parse_boundary_patch_list(&boundary_tokens)?;
    let faces = parse_face_list(faces_source_bytes)?;
    let points = parse_points_list(points_source_bytes)?;
    let face_count = faces.len() as u64;

    let start = usize::try_from(record.start_face)
        .map_err(|_| OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    let count = usize::try_from(record.n_faces)
        .map_err(|_| OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    let end = start
        .checked_add(count)
        .ok_or(OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    if end > faces.len() {
        return Err(
            OpenFoamBoundaryObservationError::BoundaryFaceRangeOutOfBounds {
                start_face: record.start_face,
                n_faces: record.n_faces,
                face_count: faces.len() as u64,
            },
        );
    }

    let internal_face_count = parse_neighbour_list_count(neighbour_source_bytes)?;

    let radius_mm = interface.radius_mm() as f64;
    let normal = interface.interface_plane.normal_unit;
    let origin = interface.interface_plane.origin_mm;
    let tolerance = tolerance_mm;
    for face in &faces[start..end] {
        for &point_index in face {
            let point = points
                .get(usize::try_from(point_index).map_err(|_| {
                    OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                        face_index: record.start_face,
                        point_index,
                    }
                })?)
                .ok_or(OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                    face_index: record.start_face,
                    point_index,
                })?;
            let point_mm = [
                point[0] * point_scale_mm_per_unit,
                point[1] * point_scale_mm_per_unit,
                point[2] * point_scale_mm_per_unit,
            ];
            let delta = [
                point_mm[0] - origin[0] as f64,
                point_mm[1] - origin[1] as f64,
                point_mm[2] - origin[2] as f64,
            ];
            let signed_distance =
                delta[0] * normal[0] as f64
                    + delta[1] * normal[1] as f64
                    + delta[2] * normal[2] as f64;
            if !signed_distance.is_finite() || signed_distance.abs() > tolerance {
                return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
            }

            let radial = [
                delta[0] - signed_distance * normal[0] as f64,
                delta[1] - signed_distance * normal[1] as f64,
                delta[2] - signed_distance * normal[2] as f64,
            ];
            let radial_distance =
                (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
            if !radial_distance.is_finite() || radial_distance > radius_mm + tolerance {
                return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
            }
        }
    }
    if internal_face_count > face_count {
        return Err(
            OpenFoamBoundaryObservationError::BoundaryPatchRangeExceedsFaces {
                patch: "<internal-faces>".to_string(),
                end_face: internal_face_count,
                face_count,
            },
        );
    }
    validate_boundary_patch_partition(
        &boundary_records,
        internal_face_count,
        face_count,
    )?;

    let mut edge_occurrences =
        std::collections::BTreeMap::<
            symthaea_passive_solver_binding::BoundaryEdgeKey,
            u32,
        >::new();
    let mut edge_origins =
        std::collections::BTreeMap::<symthaea_passive_solver_binding::BoundaryEdgeKey, (u64, u64)>::new();

    for (offset, face) in faces[start..end].iter().enumerate() {
        if face.len() < 3 {
            return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord {
                face_index: record.start_face + offset as u64,
            });
        }

        let mut seen_points = std::collections::BTreeSet::new();
        for &point_index in face {
            if !seen_points.insert(point_index) {
                return Err(OpenFoamBoundaryObservationError::InvalidFaceRecord {
                    face_index: record.start_face + offset as u64,
                });
            }
        }

        for edge_index in 0..face.len() {
            let a_index = face[edge_index];
            let b_index = face[(edge_index + 1) % face.len()];
            let a = points
                .get(usize::try_from(a_index).map_err(|_| {
                    OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                        face_index: record.start_face + offset as u64,
                        point_index: a_index,
                    }
                })?)
                .ok_or(OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                    face_index: record.start_face + offset as u64,
                    point_index: a_index,
                })?;
            let b = points
                .get(usize::try_from(b_index).map_err(|_| {
                    OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                        face_index: record.start_face + offset as u64,
                        point_index: b_index,
                    }
                })?)
                .ok_or(OpenFoamBoundaryObservationError::PointIndexOutOfBounds {
                    face_index: record.start_face + offset as u64,
                    point_index: b_index,
                })?;

            let a_mm = scaled_point_to_f32(*a, point_scale_mm_per_unit)?;
            let b_mm = scaled_point_to_f32(*b, point_scale_mm_per_unit)?;
            let edge = symthaea_passive_solver_binding::BoundaryEdgeKey::new(a_mm, b_mm)
                .map_err(|_| OpenFoamBoundaryObservationError::PatchGeometryMismatch)?;
            let topology_edge = if a_index <= b_index {
                (a_index, b_index)
            } else {
                (b_index, a_index)
            };
            if let Some(existing) = edge_origins.get(&edge) {
                if *existing != topology_edge {
                    return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
                }
            } else {
                edge_origins.insert(edge, topology_edge);
            }
            let count = edge_occurrences.entry(edge).or_insert(0);
            *count = count
                .checked_add(1)
                .ok_or(OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
            if *count > 2 {
                return Err(OpenFoamBoundaryObservationError::NonManifoldPatchEdge {
                    face_index: record.start_face + offset as u64,
                    point_a: a_index,
                    point_b: b_index,
                });
            }
        }
    }

    let solver_boundary_edges: std::collections::BTreeSet<_> = edge_occurrences
        .into_iter()
        .filter_map(|(edge, occurrences)| (occurrences == 1).then_some(edge))
        .collect();

    if solver_boundary_edges.is_empty() {
        return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
    }

    let candidate_selection =
        symthaea_passive_solver_binding::select_boundary_patch(
            interface,
            candidate,
            tolerance_mm,
        )
        .map_err(|_| OpenFoamBoundaryObservationError::PatchGeometryMismatch)?;

    let candidate_edges: std::collections::BTreeSet<_> =
        candidate_selection.edges().iter().copied().collect();
    if solver_boundary_edges != candidate_edges {
        return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
    }

    let mut source_hasher = Hasher::new();
    source_hasher.update(b"openfoam-polyMesh-patch-geometry-source:v1");
    source_hasher.update(&(boundary_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(boundary_source_bytes);
    source_hasher.update(&(faces_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(faces_source_bytes);
    source_hasher.update(&(points_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(points_source_bytes);
    source_hasher.update(&(neighbour_source_bytes.len() as u64).to_le_bytes());
    source_hasher.update(neighbour_source_bytes);
    let source_digest = *source_hasher.finalize().as_bytes();

    let mut identity = Vec::new();
    identity.extend_from_slice(b"openfoam-polyMesh-patch-geometry-with-neighbour:v1");
    let boundary_identity = record.canonical_identity_bytes();
    identity.extend_from_slice(&(boundary_identity.len() as u64).to_le_bytes());
    identity.extend_from_slice(&boundary_identity);
    identity.extend_from_slice(&point_scale_mm_per_unit.to_le_bytes());
    identity.extend_from_slice(&tolerance_mm.to_le_bytes());
    identity.extend_from_slice(&(solver_boundary_edges.len() as u64).to_le_bytes());
    for edge in &solver_boundary_edges {
        identity.extend_from_slice(&edge.a[0].to_le_bytes());
        identity.extend_from_slice(&edge.a[1].to_le_bytes());
        identity.extend_from_slice(&edge.a[2].to_le_bytes());
        identity.extend_from_slice(&edge.b[0].to_le_bytes());
        identity.extend_from_slice(&edge.b[1].to_le_bytes());
        identity.extend_from_slice(&edge.b[2].to_le_bytes());
    }

    let observation = SolverBoundaryEntityObservation::new(
        "openfoam-polyMesh-patch-geometry-with-neighbour:v1",
        identity,
        source_digest,
    )
    .map_err(Into::into)?;

    Ok((record, observation))
}

fn scaled_point_to_f32(
    point: [f64; 3],
    scale_mm_per_unit: f64,
) -> Result<[f32; 3], OpenFoamBoundaryObservationError> {
    let mut scaled = [0.0f32; 3];
    for (index, value) in point.into_iter().enumerate() {
        let value = value * scale_mm_per_unit;
        if !value.is_finite()
            || value > f32::MAX as f64
            || value < -(f32::MAX as f64)
        {
            return Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch);
        }
        scaled[index] = value as f32;
    }
    Ok(scaled)
}

fn parse_points_list(
    source_bytes: &[u8],
) -> Result<Vec<[f64; 3]>, OpenFoamBoundaryObservationError> {
    let text = std::str::from_utf8(source_bytes)
        .map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?;
    let stripped = strip_comments(text)?;
    let tokens = tokenize(&stripped)?;

    let (count, mut index) = locate_top_level_list(
        &tokens,
        OpenFoamBoundaryObservationError::MissingPointList,
        OpenFoamBoundaryObservationError::DuplicatePointList,
        OpenFoamBoundaryObservationError::InvalidPointRecord { point_index: 0 },
    )?;

    let mut points = Vec::new();
    while index < tokens.len() && !matches!(tokens[index], Token::RParen) {
        let point_index = points.len() as u64;
        if !matches!(tokens.get(index), Some(Token::LParen)) {
            return Err(OpenFoamBoundaryObservationError::InvalidPointRecord { point_index });
        }
        index += 1;
        let mut point = [0.0f64; 3];
        for coordinate in &mut point {
            let Token::Number(value) = tokens.get(index).ok_or(
                OpenFoamBoundaryObservationError::InvalidPointRecord { point_index },
            )? else {
                return Err(OpenFoamBoundaryObservationError::InvalidPointRecord { point_index });
            };
            *coordinate = value.parse::<f64>().map_err(|_| {
                OpenFoamBoundaryObservationError::InvalidPointRecord { point_index }
            })?;
            if !coordinate.is_finite() {
                return Err(OpenFoamBoundaryObservationError::InvalidPointRecord { point_index });
            }
            index += 1;
        }
        if !matches!(tokens.get(index), Some(Token::RParen)) {
            return Err(OpenFoamBoundaryObservationError::InvalidPointRecord { point_index });
        }
        index += 1;
        points.push(point);
    }

    if !matches!(tokens.get(index), Some(Token::RParen)) || index + 1 != tokens.len() {
        return Err(OpenFoamBoundaryObservationError::InvalidFaceListEntry);
    }

    let observed = points.len() as u64;
    if observed != count {
        return Err(OpenFoamBoundaryObservationError::PointsCountMismatch {
            declared: count,
            observed,
        });
    }
    Ok(points)
}

fn locate_top_level_list(
    tokens: &[Token],
    missing: OpenFoamBoundaryObservationError,
    duplicate: OpenFoamBoundaryObservationError,
    malformed: OpenFoamBoundaryObservationError,
) -> Result<(u64, usize), OpenFoamBoundaryObservationError> {
    let mut brace_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut found = None;

    for index in 0..tokens.len().saturating_sub(1) {
        if brace_depth == 0
            && paren_depth == 0
            && matches!(&tokens[index], Token::Number(_))
            && matches!(tokens[index + 1], Token::LParen)
        {
            if found.is_some() {
                return Err(duplicate);
            }
            let Token::Number(count) = &tokens[index] else {
                unreachable!();
            };
            let count = count.parse::<u64>().map_err(|_| malformed.clone())?;
            found = Some((count, index + 2));
        }

        match tokens[index] {
            Token::LBrace => brace_depth = brace_depth.checked_add(1).ok_or_else(|| malformed.clone())?,
            Token::RBrace => brace_depth = brace_depth.checked_sub(1).ok_or_else(|| malformed.clone())?,
            Token::LParen => paren_depth = paren_depth.checked_add(1).ok_or_else(|| malformed.clone())?,
            Token::RParen => paren_depth = paren_depth.checked_sub(1).ok_or_else(|| malformed.clone())?,
            Token::Number(_) | Token::Ident(_) | Token::Semi => {}
        }
    }

    if brace_depth != 0 || paren_depth != 0 {
        return Err(malformed);
    }
    found.ok_or(missing)
}

fn parse_boundary_patch_list(
    tokens: &[Token],
) -> Result<Vec<OpenFoamBoundaryPatchRecord>, OpenFoamBoundaryObservationError> {
    let mut brace_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut list_start = None;
    let mut list_count = None;

    for index in 0..tokens.len().saturating_sub(1) {
        if brace_depth == 0
            && paren_depth == 0
            && matches!(&tokens[index], Token::Number(_))
            && matches!(tokens[index + 1], Token::LParen)
        {
            if list_start.is_some() {
                return Err(OpenFoamBoundaryObservationError::DuplicatePatchList);
            }
            let Token::Number(count) = &tokens[index] else {
                unreachable!();
            };
            let declared = count.parse::<u64>().map_err(|_| {
                OpenFoamBoundaryObservationError::InvalidNumericField {
                    patch: "<boundary-list>".to_string(),
                    field: "patchCount",
                }
            })?;
            list_count = Some(declared);
            list_start = Some(index + 2);
        }

        match tokens[index] {
            Token::LBrace => {
                brace_depth = brace_depth.checked_add(1).ok_or(
                    OpenFoamBoundaryObservationError::ArithmeticOverflow,
                )?
            }
            Token::RBrace => {
                brace_depth = brace_depth.checked_sub(1).ok_or(
                    OpenFoamBoundaryObservationError::UnexpectedEndOfInput,
                )?
            }
            Token::LParen => {
                paren_depth = paren_depth.checked_add(1).ok_or(
                    OpenFoamBoundaryObservationError::ArithmeticOverflow,
                )?
            }
            Token::RParen => {
                paren_depth = paren_depth.checked_sub(1).ok_or(
                    OpenFoamBoundaryObservationError::UnexpectedEndOfInput,
                )?
            }
            Token::Number(_) | Token::Ident(_) | Token::Semi => {}
        }
    }

    if brace_depth != 0 || paren_depth != 0 {
        return Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput);
    }

    let mut index = list_start.ok_or(OpenFoamBoundaryObservationError::MissingPatchList)?;
    let declared = list_count.expect("list count set with list start");
    let mut records = Vec::new();

    while index < tokens.len() {
        if matches!(tokens[index], Token::RParen) {
            index += 1;
            break;
        }

        let Token::Ident(name) = &tokens[index] else {
            return Err(OpenFoamBoundaryObservationError::UnexpectedPatchListEntry);
        };
        if !matches!(tokens.get(index + 1), Some(Token::LBrace)) {
            return Err(OpenFoamBoundaryObservationError::UnexpectedPatchListEntry);
        }

        let record = parse_patch_block(tokens, index + 2, name)?;
        if records.iter().any(|existing: &OpenFoamBoundaryPatchRecord| existing.patch_name == record.patch_name) {
            return Err(OpenFoamBoundaryObservationError::DuplicatePatch(
                record.patch_name,
            ));
        }
        records.push(record);
        index = next_patch_block_end(tokens, index + 2)?;
    }

    if !records.iter().all(|record| !record.patch_name.trim().is_empty()) {
        return Err(OpenFoamBoundaryObservationError::UnexpectedPatchListEntry);
    }

    let observed = records.len() as u64;
    if observed != declared {
        return Err(OpenFoamBoundaryObservationError::PatchCountMismatch {
            declared,
            observed,
        });
    }

    if index == tokens.len() {
        return Ok(records);
    }

    Err(OpenFoamBoundaryObservationError::UnexpectedPatchListEntry)
}

fn next_patch_block_end(
    tokens: &[Token],
    mut index: usize,
) -> Result<usize, OpenFoamBoundaryObservationError> {
    let mut depth = 1usize;
    while index < tokens.len() {
        match tokens[index] {
            Token::LBrace => depth = depth.checked_add(1).ok_or(
                OpenFoamBoundaryObservationError::ArithmeticOverflow,
            )?,
            Token::RBrace => {
                depth = depth.checked_sub(1).ok_or(
                    OpenFoamBoundaryObservationError::UnexpectedEndOfInput,
                )?;
                if depth == 0 {
                    return Ok(index + 1);
                }
            }
            Token::Ident(_)
            | Token::Number(_)
            | Token::LParen
            | Token::RParen
            | Token::Semi => {}
        }
        index += 1;
    }

    Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Number(String),
    Quoted(String),
    LBrace,
    RBrace,
    LParen,
    RParen,
    Semi,
}

fn strip_comments(input: &str) -> Result<String, OpenFoamBoundaryObservationError> {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut index = 0;
    let mut block_depth = 0usize;
    let mut in_quote = false;
    let mut escaped = false;

    while index < bytes.len() {
        if in_quote {
            let byte = bytes[index];
            out.push(byte as char);
            index += 1;
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_quote = false;
            }
            continue;
        }

        if block_depth > 0 {
            if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'*' {
                block_depth = block_depth.checked_add(1).ok_or(
                    OpenFoamBoundaryObservationError::ArithmeticOverflow,
                )?;
                index += 2;
            } else if index + 1 < bytes.len() && bytes[index] == b'*' && bytes[index + 1] == b'/' {
                block_depth -= 1;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }

        if bytes[index] == b'"' {
            in_quote = true;
            escaped = false;
            out.push(bytes[index] as char);
            index += 1;
            continue;
        }
        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'/' {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'*' {
            block_depth = 1;
            index += 2;
            continue;
        }
        out.push(bytes[index] as char);
        index += 1;
    }

    if block_depth != 0 || in_quote {
        return Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput);
    }
    Ok(out)
}

fn tokenize(input: &str) -> Result<Vec<Token>, OpenFoamBoundaryObservationError> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_whitespace() { index += 1; continue; }
        let token = match byte {
            b'{' => { index += 1; Token::LBrace },
            b'}' => { index += 1; Token::RBrace },
            b'(' => { index += 1; Token::LParen },
            b')' => { index += 1; Token::RParen },
            b';' => { index += 1; Token::Semi },
            b'#' => return Err(OpenFoamBoundaryObservationError::UnexpectedCharacter(byte)),
            b'"' => {
                let mut value = String::new();
                index += 1;
                let mut escaped = false;
                let mut closed = false;
                while index < bytes.len() {
                    let byte = bytes[index];
                    index += 1;
                    if escaped {
                        value.push(byte as char);
                        escaped = false;
                        continue;
                    }
                    if byte == b'\\' {
                        escaped = true;
                        continue;
                    }
                    if byte == b'"' {
                        closed = true;
                        break;
                    }
                    value.push(byte as char);
                }
                if !closed || escaped {
                    return Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput);
                }
                Token::Quoted(value)
            }
            b'\'' => return Err(OpenFoamBoundaryObservationError::UnexpectedCharacter(byte)),
            _ if is_number_start(byte) => {
                let start = index; index += 1;
                while index < bytes.len() && is_number_char(bytes[index]) { index += 1; }
                Token::Number(String::from_utf8(bytes[start..index].to_vec()).map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?)
            }
            _ if is_ident_char(byte) => {
                let start = index; index += 1;
                while index < bytes.len() && is_ident_char(bytes[index]) { index += 1; }
                Token::Ident(String::from_utf8(bytes[start..index].to_vec()).map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?)
            }
            other => return Err(OpenFoamBoundaryObservationError::UnexpectedCharacter(other)),
        };
        tokens.push(token);
    }
    Ok(tokens)
}

fn is_ident_char(byte: u8) -> bool { byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'/' ) }
fn is_number_start(byte: u8) -> bool { byte.is_ascii_digit() || matches!(byte, b'+' | b'-') }
fn is_number_char(byte: u8) -> bool { byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.' | b'e' | b'E') }

fn parse_patch_block(tokens: &[Token], mut index: usize, patch_name: &str) -> Result<OpenFoamBoundaryPatchRecord, OpenFoamBoundaryObservationError> {
    let mut block_depth = 1usize;
    let mut patch_type = None;
    let mut n_faces = None;
    let mut start_face = None;

    while index < tokens.len() && block_depth > 0 {
        match &tokens[index] {
            Token::LBrace => { block_depth += 1; index += 1; }
            Token::RBrace => { block_depth -= 1; index += 1; }
            Token::Ident(field) if block_depth == 1 => {
                let field_name = field.as_str();
                index += 1;
                if field_name == "type" {
                    if patch_type.is_some() {
                        return Err(OpenFoamBoundaryObservationError::DuplicatePatchField {
                            patch: patch_name.to_string(),
                            field: "type",
                        });
                    }
                    let Token::Ident(value) = tokens.get(index).ok_or(OpenFoamBoundaryObservationError::UnexpectedEndOfInput)? else {
                        return Err(OpenFoamBoundaryObservationError::InvalidPatchType(patch_name.to_string()));
                    };
                    patch_type = Some(value.clone()); index += 1;
                } else if matches!(field_name, "nFaces" | "startFace") {
                    let Token::Number(value) = tokens.get(index).ok_or(OpenFoamBoundaryObservationError::UnexpectedEndOfInput)? else {
                        return Err(OpenFoamBoundaryObservationError::InvalidNumericField { patch: patch_name.to_string(), field: if field_name == "nFaces" { "nFaces" } else { "startFace" } });
                    };
                    let field = if field_name == "nFaces" { "nFaces" } else { "startFace" };
                    if (field_name == "nFaces" && n_faces.is_some()) || (field_name == "startFace" && start_face.is_some()) {
                        return Err(OpenFoamBoundaryObservationError::DuplicatePatchField {
                            patch: patch_name.to_string(),
                            field,
                        });
                    }
                    let parsed = value.parse::<u64>().map_err(|_| OpenFoamBoundaryObservationError::InvalidNumericField {
                        patch: patch_name.to_string(),
                        field,
                    })?;
                    if field_name == "nFaces" { n_faces = Some(parsed); } else { start_face = Some(parsed); }
                    index += 1;
                } else { skip_value(tokens, &mut index)?; }
                if matches!(tokens.get(index), Some(Token::Semi)) { index += 1; }
                else if block_depth == 1 { return Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput); }
            }
            _ => index += 1,
        }
    }

    if block_depth != 0 { return Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput); }
    let patch_type = patch_type.ok_or(OpenFoamBoundaryObservationError::MissingPatchField { patch: patch_name.to_string(), field: "type" })?;
    let n_faces = n_faces.ok_or(OpenFoamBoundaryObservationError::MissingPatchField { patch: patch_name.to_string(), field: "nFaces" })?;
    let start_face = start_face.ok_or(OpenFoamBoundaryObservationError::MissingPatchField { patch: patch_name.to_string(), field: "startFace" })?;
    if patch_type.trim().is_empty() { return Err(OpenFoamBoundaryObservationError::InvalidPatchType(patch_name.to_string())); }
    start_face.checked_add(n_faces).ok_or(OpenFoamBoundaryObservationError::ArithmeticOverflow)?;
    Ok(OpenFoamBoundaryPatchRecord { patch_name: patch_name.to_string(), patch_type, n_faces, start_face })
}

fn skip_value(tokens: &[Token], index: &mut usize) -> Result<(), OpenFoamBoundaryObservationError> {
    match tokens.get(*index) {
        Some(Token::Quoted(_)) => {
            *index += 1;
            Ok(())
        }
        Some(Token::LParen) => {
            let mut depth = 1usize; *index += 1;
            while *index < tokens.len() && depth > 0 {
                match tokens[*index] { Token::LParen => depth += 1, Token::RParen => depth -= 1, _ => {} }
                *index += 1;
            }
            if depth != 0 { return Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput); }
            Ok(())
        }
        Some(Token::LBrace) => {
            let mut depth = 1usize; *index += 1;
            while *index < tokens.len() && depth > 0 {
                match tokens[*index] { Token::LBrace => depth += 1, Token::RBrace => depth -= 1, _ => {} }
                *index += 1;
            }
            if depth != 0 { return Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput); }
            Ok(())
        }
        Some(Token::Ident(_) | Token::Number(_) | Token::Quoted(_)) => {
            *index += 1;
            Ok(())
        }
        Some(Token::RBrace | Token::RParen | Token::Semi) | None => Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput),
    }
}

/// OpenFOAM passive boundary adapter backed by exact mesh-input artifacts.
///
/// The adapter establishes solver-input entity provenance only. When a faces
/// artifact is supplied through new_with_faces, it additionally proves that the
/// boundary patch's declared face range lies within the declared global face list.
/// It still does not imply that a live solver loaded the files, that the face
/// geometry matches the candidate mesh, or that any physics ran.
pub struct OpenFoamPassiveBoundaryAdapter {
    source_bytes: Vec<u8>,
    faces_source_bytes: Option<Vec<u8>>,
    points_source_bytes: Option<Vec<u8>>,
    neighbour_source_bytes: Option<Vec<u8>>,
    owner_source_bytes: Option<Vec<u8>>,
    point_scale_mm_per_unit: Option<f64>,
    patch_name: String,
    tolerance_mm: f64,
}

impl OpenFoamPassiveBoundaryAdapter {
    fn validate_artifact_configuration(
        &self,
    ) -> Result<(), OpenFoamBoundaryObservationError> {
        let has_faces = self.faces_source_bytes.is_some();
        let has_points = self.points_source_bytes.is_some();
        let has_neighbour = self.neighbour_source_bytes.is_some();
        let has_owner = self.owner_source_bytes.is_some();
        let has_scale = self.point_scale_mm_per_unit.is_some();

        if has_points && (!has_faces || !has_scale) {
            return Err(OpenFoamBoundaryObservationError::InvalidArtifactCombination);
        }
        if has_neighbour && (!has_faces || !has_points || !has_scale) {
            return Err(OpenFoamBoundaryObservationError::InvalidArtifactCombination);
        }
        if has_owner && (!has_faces || !has_points || !has_neighbour || !has_scale) {
            return Err(OpenFoamBoundaryObservationError::InvalidArtifactCombination);
        }
        if has_scale && !has_points {
            return Err(OpenFoamBoundaryObservationError::InvalidArtifactCombination);
        }
        Ok(())
    }

    pub fn new(
        source_bytes: impl Into<Vec<u8>>,
        patch_name: impl Into<String>,
        tolerance_mm: f64,
    ) -> Result<Self, OpenFoamBoundaryObservationError> {
        let patch_name = patch_name.into();
        if patch_name.trim().is_empty() {
            return Err(OpenFoamBoundaryObservationError::PatchNotFound(patch_name));
        }
        if !tolerance_mm.is_finite() || tolerance_mm < 0.0 {
            return Err(OpenFoamBoundaryObservationError::InvalidTolerance);
        }
        Ok(Self {
            source_bytes: source_bytes.into(),
            faces_source_bytes: None,
            points_source_bytes: None,
            neighbour_source_bytes: None,
            owner_source_bytes: None,
            point_scale_mm_per_unit: None,
            patch_name,
            tolerance_mm,
        })
    }

    /// Construct an input-evidence adapter that also checks the patch range
    /// against the exact serialized constant/polyMesh/faces artifact.
    pub fn new_with_faces(
        source_bytes: impl Into<Vec<u8>>,
        faces_source_bytes: impl Into<Vec<u8>>,
        patch_name: impl Into<String>,
        tolerance_mm: f64,
    ) -> Result<Self, OpenFoamBoundaryObservationError> {
        let mut adapter = Self::new(source_bytes, patch_name, tolerance_mm)?;
        adapter.faces_source_bytes = Some(faces_source_bytes.into());
        Ok(adapter)
    }

    /// Construct an input-evidence adapter that also requires the rendered
    /// OpenFOAM patch perimeter to match the candidate's certified rim.
    pub fn new_with_faces_and_points(
        source_bytes: impl Into<Vec<u8>>,
        faces_source_bytes: impl Into<Vec<u8>>,
        points_source_bytes: impl Into<Vec<u8>>,
        point_scale_mm_per_unit: f64,
        patch_name: impl Into<String>,
        tolerance_mm: f64,
    ) -> Result<Self, OpenFoamBoundaryObservationError> {
        let mut adapter = Self::new_with_faces(
            source_bytes,
            faces_source_bytes,
            patch_name,
            tolerance_mm,
        )?;
        if !point_scale_mm_per_unit.is_finite() || point_scale_mm_per_unit <= 0.0 {
            return Err(OpenFoamBoundaryObservationError::InvalidPointScale);
        }
        adapter.points_source_bytes = Some(points_source_bytes.into());
        adapter.point_scale_mm_per_unit = Some(point_scale_mm_per_unit);
        Ok(adapter)
    }

    /// Construct an input-evidence adapter that additionally validates the
    /// complete boundary partition against the exact neighbour list.
    pub fn new_with_faces_points_and_neighbour(
        source_bytes: impl Into<Vec<u8>>,
        faces_source_bytes: impl Into<Vec<u8>>,
        points_source_bytes: impl Into<Vec<u8>>,
        neighbour_source_bytes: impl Into<Vec<u8>>,
        point_scale_mm_per_unit: f64,
        patch_name: impl Into<String>,
        tolerance_mm: f64,
    ) -> Result<Self, OpenFoamBoundaryObservationError> {
        let mut adapter = Self::new_with_faces_and_points(
            source_bytes,
            faces_source_bytes,
            points_source_bytes,
            point_scale_mm_per_unit,
            patch_name,
            tolerance_mm,
        )?;
        adapter.neighbour_source_bytes = Some(neighbour_source_bytes.into());
        Ok(adapter)
    }

    /// Construct an input-evidence adapter that also cross-checks the exact
    /// owner cardinality against the global face list and rejects internal-face
    /// owner/neighbour self-loops.
    pub fn new_with_complete_mesh_topology(
        source_bytes: impl Into<Vec<u8>>,
        faces_source_bytes: impl Into<Vec<u8>>,
        points_source_bytes: impl Into<Vec<u8>>,
        neighbour_source_bytes: impl Into<Vec<u8>>,
        owner_source_bytes: impl Into<Vec<u8>>,
        point_scale_mm_per_unit: f64,
        patch_name: impl Into<String>,
        tolerance_mm: f64,
    ) -> Result<Self, OpenFoamBoundaryObservationError> {
        let mut adapter = Self::new_with_faces_points_and_neighbour(
            source_bytes,
            faces_source_bytes,
            points_source_bytes,
            neighbour_source_bytes,
            point_scale_mm_per_unit,
            patch_name,
            tolerance_mm,
        )?;
        adapter.owner_source_bytes = Some(owner_source_bytes.into());
        Ok(adapter)
    }
}

impl symthaea_passive_solver_binding::SolverBoundaryBindingAdapter
    for OpenFoamPassiveBoundaryAdapter
{
    fn adapter_id(&self) -> &str {
        "openfoam-passive-boundary/v1"
    }

    fn bind(
        &self,
        interface: &symthaea_passive_void_compiler::PortInterface,
        candidate: &symthaea_fabrication_kernel::mesh::TriangleMesh,
        _candidate_geometry_digest: [u8; 32],
    ) -> Result<
        symthaea_passive_solver_binding::SolverBoundaryBindingDraft,
        symthaea_passive_solver_binding::SolverBindingError,
    > {
        let boundary_patch =
            symthaea_passive_solver_binding::select_boundary_patch(
                interface,
                candidate,
                self.tolerance_mm,
            )?;
        symthaea_passive_solver_binding::SolverBoundaryBindingDraft::new(
            format!("openfoam:patch:{}", self.patch_name),
            boundary_patch,
        )
    }
}

impl symthaea_passive_solver_binding::SolverBoundaryInputEntityObserver
    for OpenFoamPassiveBoundaryAdapter
{
    fn observe_input_entity(
        &self,
        interface: &symthaea_passive_void_compiler::PortInterface,
        candidate: &symthaea_fabrication_kernel::mesh::TriangleMesh,
        binding: &symthaea_passive_solver_binding::SolverBoundaryBinding,
    ) -> Result<
        symthaea_passive_solver_binding::SolverBoundaryEntityAttestation,
        symthaea_passive_solver_binding::SolverBindingError,
    > {
        self.validate_artifact_configuration().map_err(|error| {
            symthaea_passive_solver_binding::SolverBindingError::ExternalObservation(
                format!("{error:?}"),
            )
        })?;

        let (_, observation) = match (
            &self.faces_source_bytes,
            &self.points_source_bytes,
            &self.neighbour_source_bytes,
            self.point_scale_mm_per_unit,
        ) {
            (
                Some(faces_source_bytes),
                Some(points_source_bytes),
                Some(neighbour_source_bytes),
                Some(point_scale_mm_per_unit),
            ) if self.owner_source_bytes.is_none() => observe_openfoam_patch_geometry_with_neighbour(
                &self.source_bytes,
                faces_source_bytes,
                points_source_bytes,
                neighbour_source_bytes,
                &self.patch_name,
                interface,
                candidate,
                self.tolerance_mm,
                point_scale_mm_per_unit,
            ),
            (
                Some(faces_source_bytes),
                Some(points_source_bytes),
                Some(neighbour_source_bytes),
                Some(point_scale_mm_per_unit),
            ) => observe_openfoam_patch_geometry_with_owner_and_neighbour(
                &self.source_bytes,
                faces_source_bytes,
                points_source_bytes,
                neighbour_source_bytes,
                self.owner_source_bytes.as_ref().expect("owner present"),
                &self.patch_name,
                interface,
                candidate,
                self.tolerance_mm,
                point_scale_mm_per_unit,
            ),
            (Some(faces_source_bytes), Some(points_source_bytes), None, Some(point_scale_mm_per_unit)) =>
                observe_openfoam_patch_geometry(
                    &self.source_bytes,
                    faces_source_bytes,
                    points_source_bytes,
                    &self.patch_name,
                    interface,
                    candidate,
                    self.tolerance_mm,
                    point_scale_mm_per_unit,
                ),
            (Some(faces_source_bytes), _, _, _) => observe_openfoam_boundary_patch_and_faces(
                &self.source_bytes,
                faces_source_bytes,
                &self.patch_name,
            ),
            _ => observe_openfoam_boundary_patch(
                &self.source_bytes,
                &self.patch_name,
            ),
        }
        .map_err(|error| {
            symthaea_passive_solver_binding::SolverBindingError::ExternalObservation(
                format!("{error:?}"),
            )
        })?;

        let mapping_digest =
            symthaea_passive_solver_binding::solver_entity_mapping_digest(
                interface,
                &binding.adapter_id,
                binding.realized_boundary.candidate_geometry_digest(),
                symthaea_passive_solver_binding::digest_triangle_mesh(candidate),
                binding.realized_boundary.boundary_patch_digest(),
                &binding.external_boundary_handle,
                observation.fingerprint(),
                observation.digest(),
            );

        symthaea_passive_solver_binding::SolverBoundaryEntityAttestation::new(
            binding.external_boundary_handle.clone(),
            observation,
            mapping_digest,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDARY: &[u8] = br#"FoamFile
{
    version 2.0;
    format ascii;
}
2
(
    inlet
    {
        type patch;
        nFaces 4;
        startFace 10;
        inGroups (inletGroup);
    }
    outlet
    {
        type patch;
        nFaces 6;
        startFace 14;
    }
)
"#;

    #[test]
    fn input_adapter_produces_structured_solver_input_evidence() {
        use symthaea_fabrication_kernel::mesh::TriangleMesh;
        use symthaea_passive_solver_binding::{
            bind_with_adapter_and_input_entity_attestation,
            SolverBoundaryEvidenceLevel,
        };
        use symthaea_passive_void_compiler::{
            BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
            SolverBoundaryIdentity,
        };
        use symthaea_passive_void_graph::PortId;

        let interface = PortInterface::new(
            PortId(10),
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: 7,
            },
        )
        .unwrap();

        let candidate = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
            ],
        };
        let adapter =
            OpenFoamPassiveBoundaryAdapter::new(BOUNDARY.to_vec(), "inlet", 0.05)
                .unwrap();

        let binding = bind_with_adapter_and_input_entity_attestation(
            &adapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        assert_eq!(
            binding.evidence_level(),
            SolverBoundaryEvidenceLevel::SolverInputEntityAttested
        );
        assert_eq!(
            binding.external_boundary_handle,
            "openfoam:patch:inlet"
        );
        assert_eq!(
            binding.solver_entity_observation_kind(),
            Some("openfoam-polyMesh-boundary-patch:v1")
        );
    }

    #[test]
    fn comment_markers_inside_quoted_values_are_not_comments() {
        let source = br#"1
(
    inlet
    {
        type patch;
        nFaces 1;
        startFace 0;
        note "url=http://example.test/a/*literal*/";
    }
)
"#;
        let (record, _) = observe_openfoam_boundary_patch(source, "inlet").unwrap();
        assert_eq!(record.patch_name, "inlet");
        assert_eq!(record.patch_type, "patch");
    }

    #[test]
    fn real_openfoam_style_quoted_header_is_accepted() {
        let source = br#"FoamFile
{
    version 2.0;
    format ascii;
    class polyBoundaryMesh;
    location "constant/polyMesh";
    object boundary;
}
1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
// ************************************************************************* //
"#;
        let (record, _) = observe_openfoam_boundary_patch(source, "inlet").unwrap();
        assert_eq!(record.patch_name, "inlet");
        assert_eq!(record.patch_type, "patch");
    }

    #[test]
    fn parses_requested_patch_and_derives_observation() {
        let (record, observation) = observe_openfoam_boundary_patch(BOUNDARY, "inlet").unwrap();
        assert_eq!(record, OpenFoamBoundaryPatchRecord { patch_name: "inlet".into(), patch_type: "patch".into(), n_faces: 4, start_face: 10 });
        assert_eq!(observation.entity_kind, "openfoam-polyMesh-boundary-patch:v1");
        assert_ne!(observation.source_digest, [0; 32]);
    }

    #[test]
    fn duplicate_patch_names_fail_closed() {
        let source = br#"2 ( inlet { type patch; nFaces 1; startFace 0; } inlet { type patch; nFaces 1; startFace 1; } )"#;
        assert!(matches!(observe_openfoam_boundary_patch(source, "inlet"), Err(OpenFoamBoundaryObservationError::DuplicatePatch(_))));
    }

    #[test]
    fn duplicate_patch_fields_fail_closed() {
        let duplicate_type = br#"1 ( inlet { type patch; type wall; nFaces 1; startFace 0; } )"#;
        assert!(matches!(
            observe_openfoam_boundary_patch(duplicate_type, "inlet"),
            Err(OpenFoamBoundaryObservationError::DuplicatePatchField { field: "type", .. })
        ));

        let duplicate_n_faces = br#"1 ( inlet { type patch; nFaces 1; nFaces 2; startFace 0; } )"#;
        assert!(matches!(
            observe_openfoam_boundary_patch(duplicate_n_faces, "inlet"),
            Err(OpenFoamBoundaryObservationError::DuplicatePatchField { field: "nFaces", .. })
        ));

        let duplicate_start_face = br#"1 ( inlet { type patch; nFaces 1; startFace 0; startFace 1; } )"#;
        assert!(matches!(
            observe_openfoam_boundary_patch(duplicate_start_face, "inlet"),
            Err(OpenFoamBoundaryObservationError::DuplicatePatchField { field: "startFace", .. })
        ));
    }

    #[test]
    fn missing_face_range_field_fails_closed() {
        let source = br#"1 ( inlet { type patch; nFaces 1; } )"#;
        assert!(matches!(observe_openfoam_boundary_patch(source, "inlet"), Err(OpenFoamBoundaryObservationError::MissingPatchField { field: "startFace", .. })));
    }

    #[test]
    fn source_bytes_are_part_of_observation_lineage() {
        let (_, first) = observe_openfoam_boundary_patch(BOUNDARY, "inlet").unwrap();
        let mut changed = BOUNDARY.to_vec(); changed.extend_from_slice(b"\n");
        let (_, second) = observe_openfoam_boundary_patch(&changed, "inlet").unwrap();
        assert_ne!(first.source_digest, second.source_digest);
        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn declared_patch_count_must_match_observed_entries() {
        let source = br#"3
(
    inlet { type patch; nFaces 1; startFace 0; }
    outlet { type patch; nFaces 1; startFace 1; }
)
"#;
        assert!(matches!(
            observe_openfoam_boundary_patch(source, "inlet"),
            Err(OpenFoamBoundaryObservationError::PatchCountMismatch {
                declared: 3,
                observed: 2
            })
        ));
    }

    #[test]
    fn nested_same_named_block_cannot_shadow_boundary_patch() {
        let source = br#"1
(
    inlet
    {
        type patch;
        nFaces 1;
        startFace 0;
        nested
        {
            inlet { type fake; nFaces 9; startFace 9; }
        }
    }
)
"#;
        let (record, _) = observe_openfoam_boundary_patch(source, "inlet").unwrap();
        assert_eq!(record.patch_type, "patch");
        assert_eq!(record.n_faces, 1);
        assert_eq!(record.start_face, 0);
    }

    #[test]
    fn boundary_range_must_fit_within_faces_list() {
        let faces = br#"3
(
    3(0 1 2)
    3(2 3 4)
    3(4 5 6)
)
"#;
        let boundary = br#"1
(
    inlet { type patch; nFaces 2; startFace 1; }
)
"#;
        let (_, observation) =
            observe_openfoam_boundary_patch_and_faces(boundary, faces, "inlet")
                .unwrap();
        assert_eq!(
            observation.entity_kind,
            "openfoam-polyMesh-boundary-patch-and-face-list:v1"
        );

        let out_of_bounds = br#"1
(
    inlet { type patch; nFaces 2; startFace 2; }
)
"#;
        assert!(matches!(
            observe_openfoam_boundary_patch_and_faces(&out_of_bounds, faces, "inlet"),
            Err(OpenFoamBoundaryObservationError::BoundaryFaceRangeOutOfBounds {
                start_face: 2,
                n_faces: 2,
                face_count: 3
            })
        ));
    }

    #[test]
    fn face_source_changes_change_combined_observation_lineage() {
        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    3(0 1 2)
)
"#;
        let changed_faces = br#"1
(
    3(0 2 1)
)
"#;
        let (_, first) =
            observe_openfoam_boundary_patch_and_faces(boundary, faces, "inlet").unwrap();
        let (_, second) =
            observe_openfoam_boundary_patch_and_faces(boundary, changed_faces, "inlet").unwrap();
        assert_ne!(first.source_digest, second.source_digest);
        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn patch_geometry_with_neighbour_must_match_candidate_rim() {
        use symthaea_fabrication_kernel::mesh::TriangleMesh;
        use symthaea_passive_void_compiler::{
            BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
            SolverBoundaryIdentity,
        };
        use symthaea_passive_void_graph::PortId;

        let interface = PortInterface::new(
            PortId(10),
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: 7,
            },
        )
        .unwrap();
        let candidate = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
        };

        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    4(0 1 2 3)
)
"#;
        let points = br#"4
(
    (2 0 0)
    (0 2 0)
    (-2 0 0)
    (0 -2 0)
)
"#;

        let neighbour = br#"0
(
)
"#;
        let (_, observation) = observe_openfoam_patch_geometry_with_neighbour(
            boundary, faces, points, neighbour, "inlet", &interface, &candidate, 0.05, 1.0,
        ).unwrap();
        assert_eq!(
            observation.entity_kind,
            "openfoam-polyMesh-patch-geometry-with-neighbour:v1"
        );

        let adapter = OpenFoamPassiveBoundaryAdapter::new_with_faces_points_and_neighbour(
            boundary.to_vec(),
            faces.to_vec(),
            points.to_vec(),
            br#"0
(
)
"#.to_vec(),
            1.0,
            "inlet",
            0.05,
        )
        .unwrap();
        let binding =
            symthaea_passive_solver_binding::bind_with_adapter_and_input_entity_attestation(
                &adapter,
                &interface,
                &candidate,
                [7; 32],
                0.05,
            )
            .unwrap();
        assert_eq!(
            binding.evidence_level(),
            symthaea_passive_solver_binding::SolverBoundaryEvidenceLevel::SolverInputEntityAttested
        );
        assert_eq!(
            binding.solver_entity_observation_kind(),
            Some("openfoam-polyMesh-patch-geometry-with-neighbour:v1")
        );

        let changed_points = br#"4
(
    (2 0 0)
    (0 2 0)
    (-1 0 0)
    (0 -2 0)
)
"#;
        assert!(matches!(
            observe_openfoam_patch_geometry_with_neighbour(
                boundary, faces, changed_points, neighbour, "inlet", &interface, &candidate, 0.05, 1.0
            ),
            Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch)
        ));

    }

    #[test]
    fn boundary_partition_rejects_overlap_and_gaps() {
        let overlap = vec![
            OpenFoamBoundaryPatchRecord {
                patch_name: "inlet".into(),
                patch_type: "patch".into(),
                n_faces: 2,
                start_face: 0,
            },
            OpenFoamBoundaryPatchRecord {
                patch_name: "outlet".into(),
                patch_type: "patch".into(),
                n_faces: 2,
                start_face: 1,
            },
        ];
        assert!(matches!(
            validate_boundary_patch_partition(&overlap, 0, 3),
            Err(OpenFoamBoundaryObservationError::BoundaryPatchRangeOverlap {
                ref patch
            }) if patch == "outlet"
        ));

        let gap = vec![
            OpenFoamBoundaryPatchRecord {
                patch_name: "inlet".into(),
                patch_type: "patch".into(),
                n_faces: 1,
                start_face: 0,
            },
            OpenFoamBoundaryPatchRecord {
                patch_name: "outlet".into(),
                patch_type: "patch".into(),
                n_faces: 1,
                start_face: 2,
            },
        ];
        assert!(matches!(
            validate_boundary_patch_partition(&gap, 0, 3),
            Err(OpenFoamBoundaryObservationError::NonContiguousBoundaryPatchRange {
                ref patch,
                expected_start: 1,
                actual_start: 2,
            }) if patch == "outlet"
        ));
    }

    #[test]
    fn patch_vertex_must_stay_on_interface_plane_and_inside_aperture() {
        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    4(0 1 2 3)
)
"#;
        let good_points = br#"4
(
    (2 0 0)
    (0 2 0)
    (-2 0 0)
    (0 -2 0)
)
"#;
        let bad_points = br#"4
(
    (2 0 0)
    (0 2 0)
    (-2 0 0)
    (0 -2 0.2)
)
"#;
        let interface = {
            use symthaea_passive_void_compiler::{
                BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
                SolverBoundaryIdentity,
            };
            use symthaea_passive_void_graph::PortId;
            PortInterface::new(
                PortId(10),
                [0.0, 0.0, 0.0],
                PortAperture::Circular { radius_mm: 2.0 },
                [0.0, 0.0, 1.0],
                InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
                SolverBoundaryIdentity {
                    domain: BoundaryConditionDomain::Fluidic,
                    id: 7,
                },
            )
            .unwrap()
        };
        let candidate = symthaea_fabrication_kernel::mesh::TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
        };

        let neighbour = br#"0
(
)
"#;
        observe_openfoam_patch_geometry_with_neighbour(
            boundary, faces, good_points, neighbour, "inlet",
            &interface, &candidate, 0.05, 1.0,
        ).unwrap();

        assert!(matches!(
            observe_openfoam_patch_geometry_with_neighbour(
                boundary, faces, bad_points, neighbour, "inlet",
                &interface, &candidate, 0.05, 1.0,
            ),
            Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch)
        ));
    }

    #[test]
    fn reversed_openfoam_face_orientation_fails_closed() {
        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    4(0 3 2 1)
)
"#;
        let points = br#"4
(
    (2 0 0)
    (0 2 0)
    (-2 0 0)
    (0 -2 0)
)
"#;
        let neighbour = br#"0
(
)
"#;
        let interface = {
            use symthaea_passive_void_compiler::{
                BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
                SolverBoundaryIdentity,
            };
            use symthaea_passive_void_graph::PortId;
            PortInterface::new(
                PortId(10),
                [0.0, 0.0, 0.0],
                PortAperture::Circular { radius_mm: 2.0 },
                [0.0, 0.0, 1.0],
                InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
                SolverBoundaryIdentity {
                    domain: BoundaryConditionDomain::Fluidic,
                    id: 7,
                },
            ).unwrap()
        };
        let candidate = symthaea_fabrication_kernel::mesh::TriangleMesh {
            vertices: vec![
                [0.0,0.0,0.0],[2.0,0.0,0.0],[0.0,2.0,0.0],[-2.0,0.0,0.0],[0.0,-2.0,0.0]
            ],
            normals: vec![[0.0,0.0,1.0];5],
            indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
        };
        assert!(matches!(
            observe_openfoam_patch_geometry_with_neighbour(
                boundary, faces, points, neighbour, "inlet",
                &interface, &candidate, 0.05, 1.0
            ),
            Err(OpenFoamBoundaryObservationError::FaceNormalMismatch { face_index: 0 })
        ));
    }

    #[test]
    fn repeated_point_index_in_face_fails_closed() {
        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    4(0 1 1 2)
)
"#;
        let points = br#"3
(
    (0 0 0)
    (2 0 0)
    (0 2 0)
)
"#;
        let interface = {
            use symthaea_passive_void_compiler::{
                BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
                SolverBoundaryIdentity,
            };
            use symthaea_passive_void_graph::PortId;
            PortInterface::new(
                PortId(10),
                [0.0, 0.0, 0.0],
                PortAperture::Circular { radius_mm: 2.0 },
                [0.0, 0.0, 1.0],
                InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
                SolverBoundaryIdentity {
                    domain: BoundaryConditionDomain::Fluidic,
                    id: 7,
                },
            )
            .unwrap()
        };
        let candidate = symthaea_fabrication_kernel::mesh::TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
        };
        assert!(matches!(
            observe_openfoam_patch_geometry(
                boundary, faces, points, "inlet", &interface, &candidate, 0.05, 1.0
            ),
            Err(OpenFoamBoundaryObservationError::InvalidFaceRecord { face_index: 0 })
        ));
    }

    #[test]
    fn degenerate_face_fails_closed() {
        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    3(0 1 2)
)
"#;
        let points = br#"3
(
    (0 0 0)
    (1 0 0)
    (2 0 0)
)
"#;
        let interface = {
            use symthaea_passive_void_compiler::{
                BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
                SolverBoundaryIdentity,
            };
            use symthaea_passive_void_graph::PortId;
            PortInterface::new(
                PortId(10),
                [0.0, 0.0, 0.0],
                PortAperture::Circular { radius_mm: 2.0 },
                [0.0, 0.0, 1.0],
                InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
                SolverBoundaryIdentity {
                    domain: BoundaryConditionDomain::Fluidic,
                    id: 7,
                },
            )
            .unwrap()
        };
        let candidate = symthaea_fabrication_kernel::mesh::TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
        };
        let neighbour = br#"0
(
)
"#;
        assert!(matches!(
            observe_openfoam_patch_geometry_with_neighbour(
                boundary, faces, points, neighbour, "inlet",
                &interface, &candidate, 0.05, 1.0
            ),
            Err(OpenFoamBoundaryObservationError::DegenerateFace { face_index: 0 })
        ));
    }

    #[test]
    fn geometry_observation_identity_commits_to_tolerance() {
        use symthaea_fabrication_kernel::mesh::TriangleMesh;
        use symthaea_passive_void_compiler::{
            BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
            SolverBoundaryIdentity,
        };
        use symthaea_passive_void_graph::PortId;

        let interface = PortInterface::new(
            PortId(10),
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: 7,
            },
        )
        .unwrap();
        let candidate = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
        };
        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    4(0 1 2 3)
)
"#;
        let points = br#"4
(
    (2 0 0)
    (0 2 0)
    (-2 0 0)
    (0 -2 0)
)
"#;

        let (_, first) = observe_openfoam_patch_geometry(
            boundary, faces, points, "inlet", &interface, &candidate, 0.05, 1.0,
        ).unwrap();
        let (_, second) = observe_openfoam_patch_geometry(
            boundary, faces, points, "inlet", &interface, &candidate, 0.10, 1.0,
        ).unwrap();

        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn zero_tolerance_is_not_silently_widened() {
        use symthaea_fabrication_kernel::mesh::TriangleMesh;
        use symthaea_passive_void_compiler::{
            BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
            SolverBoundaryIdentity,
        };
        use symthaea_passive_void_graph::PortId;

        let interface = PortInterface::new(
            PortId(10),
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: 7,
            },
        )
        .unwrap();

        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    4(0 1 2 3)
)
"#;
        let points = br#"4
(
    (2 0 0.0005)
    (0 2 0.0005)
    (-2 0 0.0005)
    (0 -2 0.0005)
)
"#;
        let candidate = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0005],
                [2.0, 0.0, 0.0005],
                [0.0, 2.0, 0.0005],
                [-2.0, 0.0, 0.0005],
                [0.0, -2.0, 0.0005],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
        };

        assert!(matches!(
            observe_openfoam_patch_geometry(
                boundary, faces, points, "inlet", &interface, &candidate, 0.0, 1.0,
            ),
            Err(OpenFoamBoundaryObservationError::PatchGeometryMismatch)
        ));

        assert!(observe_openfoam_patch_geometry(
            boundary, faces, points, "inlet", &interface, &candidate, 0.001, 1.0,
        ).is_ok());
    }

    #[test]
    fn internal_face_self_loop_fails_closed() {
        let owners = [3, 7];
        let neighbours = [3, 8];
        assert!(matches!(
            validate_internal_face_cells(&owners, &neighbours),
            Err(OpenFoamBoundaryObservationError::InternalFaceSelfLoop {
                face_index: 0,
                cell: 3
            })
        ));

        let valid_neighbours = [4, 8];
        assert!(validate_internal_face_cells(&owners, &valid_neighbours).is_ok());
    }

    #[test]
    fn negative_neighbour_label_fails_closed() {
        let source = br#"-1
(
    -3
)
"#;
        assert!(matches!(
            parse_neighbour_list_count(source),
            Err(OpenFoamBoundaryObservationError::InvalidNeighbourListEntry)
        ));
    }

    #[test]
    fn inconsistent_optional_artifact_combinations_fail_closed() {
        let mut adapter =
            OpenFoamPassiveBoundaryAdapter::new(
                BOUNDARY.to_vec(),
                "inlet",
                0.05,
            )
            .unwrap();
        adapter.points_source_bytes = Some(Vec::new());
        assert_eq!(
            adapter.validate_artifact_configuration(),
            Err(OpenFoamBoundaryObservationError::InvalidArtifactCombination)
        );

        let mut adapter =
            OpenFoamPassiveBoundaryAdapter::new(
                BOUNDARY.to_vec(),
                "inlet",
                0.05,
            )
            .unwrap();
        adapter.owner_source_bytes = Some(Vec::new());
        assert_eq!(
            adapter.validate_artifact_configuration(),
            Err(OpenFoamBoundaryObservationError::InvalidArtifactCombination)
        );
    }

    #[test]
    fn complete_mesh_topology_requires_owner_count_to_match_faces() {
        let boundary = br#"1
(
    inlet { type patch; nFaces 1; startFace 0; }
)
"#;
        let faces = br#"1
(
    4(0 1 2 3)
)
"#;
        let points = br#"4
(
    (2 0 0)
    (0 2 0)
    (-2 0 0)
    (0 -2 0)
)
"#;
        let neighbour = br#"0
(
)
"#;
        let owner = br#"0
(
)
"#;

        let result = parse_owner_list_count(owner);
        assert!(matches!(
            result,
            Ok(0)
        ));
        assert!(matches!(
            observe_openfoam_patch_geometry_with_owner_and_neighbour(
                boundary,
                faces,
                points,
                neighbour,
                owner,
                "inlet",
                &{
                    use symthaea_passive_void_compiler::{
                        BoundaryConditionDomain, InterfacePlane, PortAperture, PortInterface,
                        SolverBoundaryIdentity,
                    };
                    use symthaea_passive_void_graph::PortId;
                    PortInterface::new(
                        PortId(10),
                        [0.0, 0.0, 0.0],
                        PortAperture::Circular { radius_mm: 2.0 },
                        [0.0, 0.0, 1.0],
                        InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
                        SolverBoundaryIdentity {
                            domain: BoundaryConditionDomain::Fluidic,
                            id: 7,
                        },
                    ).unwrap()
                },
                &symthaea_fabrication_kernel::mesh::TriangleMesh {
                    vertices: vec![
                        [0.0,0.0,0.0],[2.0,0.0,0.0],[0.0,2.0,0.0],[-2.0,0.0,0.0],[0.0,-2.0,0.0]
                    ],
                    normals: vec![[0.0,0.0,1.0];5],
                    indices: vec![[0,1,2],[0,2,3],[0,3,4],[0,4,1]],
                },
                0.05,
                1.0
            ),
            Err(OpenFoamBoundaryObservationError::OwnerCountMismatch {
                declared: 0,
                observed: 1
            })
        ));
    }

    #[test]
    fn malformed_comment_fails_closed() {
        let source = br#"/* never closed 1 ( inlet { type patch; nFaces 1; startFace 0; } )"#;
        assert!(matches!(observe_openfoam_boundary_patch(source, "inlet"), Err(OpenFoamBoundaryObservationError::UnterminatedBlockComment)));
    }
}
