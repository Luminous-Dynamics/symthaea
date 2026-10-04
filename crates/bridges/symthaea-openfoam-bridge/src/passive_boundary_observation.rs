// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative observation of an OpenFOAM constant/polyMesh/boundary artifact.
//!
//! This module proves only that the supplied boundary-file bytes contain one
//! unambiguous patch record. It does not claim that a live solver loaded the
//! file, that the solver mesh matches the candidate mesh, or that any physics ran.

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
    PatchNotFound(String),
    MissingPatchField { patch: String, field: &'static str },
    InvalidNumericField { patch: String, field: &'static str },
    InvalidPatchType(String),
    ArithmeticOverflow,
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
) -> Result<(OpenFoamBoundaryPatchRecord, SolverBoundaryEntityObservation), OpenFoamBoundaryObservationError> {
    let text = std::str::from_utf8(source_bytes)
        .map_err(|_| OpenFoamBoundaryObservationError::InvalidUtf8)?;
    let stripped = strip_comments(text)?;
    let tokens = tokenize(&stripped)?;
    let mut matches = Vec::new();

    for index in 0..tokens.len().saturating_sub(1) {
        let Token::Ident(name) = &tokens[index] else { continue; };
        if name != patch_name || !matches!(tokens[index + 1], Token::LBrace) { continue; }
        matches.push(parse_patch_block(&tokens, index + 2, patch_name)?);
    }

    match matches.as_slice() {
        [] => Err(OpenFoamBoundaryObservationError::PatchNotFound(patch_name.to_string())),
        [record] => Ok((record.clone(), record.observe(source_bytes)?)),
        _ => Err(OpenFoamBoundaryObservationError::DuplicatePatch(patch_name.to_string())),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Number(String),
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

    while index < bytes.len() {
        if block_depth > 0 {
            if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'*' {
                block_depth += 1; index += 2;
            } else if index + 1 < bytes.len() && bytes[index] == b'*' && bytes[index + 1] == b'/' {
                block_depth -= 1; index += 2;
            } else { index += 1; }
            continue;
        }

        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'/' {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' { index += 1; }
            continue;
        }
        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'*' {
            block_depth = 1; index += 2; continue;
        }
        out.push(bytes[index] as char);
        index += 1;
    }

    if block_depth != 0 { return Err(OpenFoamBoundaryObservationError::UnterminatedBlockComment); }
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
            b'#' | b'"' | b'\'' => return Err(OpenFoamBoundaryObservationError::UnexpectedCharacter(byte)),
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
                    let Token::Ident(value) = tokens.get(index).ok_or(OpenFoamBoundaryObservationError::UnexpectedEndOfInput)? else {
                        return Err(OpenFoamBoundaryObservationError::InvalidPatchType(patch_name.to_string()));
                    };
                    patch_type = Some(value.clone()); index += 1;
                } else if matches!(field_name, "nFaces" | "startFace") {
                    let Token::Number(value) = tokens.get(index).ok_or(OpenFoamBoundaryObservationError::UnexpectedEndOfInput)? else {
                        return Err(OpenFoamBoundaryObservationError::InvalidNumericField { patch: patch_name.to_string(), field: if field_name == "nFaces" { "nFaces" } else { "startFace" } });
                    };
                    let parsed = value.parse::<u64>().map_err(|_| OpenFoamBoundaryObservationError::InvalidNumericField { patch: patch_name.to_string(), field: if field_name == "nFaces" { "nFaces" } else { "startFace" } })?;
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
        Some(Token::Ident(_) | Token::Number(_)) => { *index += 1; Ok(()) }
        Some(Token::RBrace | Token::RParen | Token::Semi) | None => Err(OpenFoamBoundaryObservationError::UnexpectedEndOfInput),
    }
}

/// OpenFOAM passive boundary adapter backed by an exact boundary-file artifact.
///
/// The adapter establishes solver-input entity provenance only. It does not imply
/// that a live solver loaded this file or that its mesh topology matches the
/// candidate mesh.
pub struct OpenFoamPassiveBoundaryAdapter {
    source_bytes: Vec<u8>,
    patch_name: String,
    tolerance_mm: f64,
}

impl OpenFoamPassiveBoundaryAdapter {
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
            patch_name,
            tolerance_mm,
        })
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
        let (_, observation) = observe_openfoam_boundary_patch(
            &self.source_bytes,
            &self.patch_name,
        )
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
    fn malformed_comment_fails_closed() {
        let source = br#"/* never closed 1 ( inlet { type patch; nFaces 1; startFace 0; } )"#;
        assert!(matches!(observe_openfoam_boundary_patch(source, "inlet"), Err(OpenFoamBoundaryObservationError::UnterminatedBlockComment)));
    }
}
