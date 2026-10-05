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
    InvalidPointScale,
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
            faces_source_bytes: None,
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
        let (_, observation) = match &self.faces_source_bytes {
            Some(faces_source_bytes) => observe_openfoam_boundary_patch_and_faces(
                &self.source_bytes,
                faces_source_bytes,
                &self.patch_name,
            ),
            None => observe_openfoam_boundary_patch(
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
    fn malformed_comment_fails_closed() {
        let source = br#"/* never closed 1 ( inlet { type patch; nFaces 1; startFace 0; } )"#;
        assert!(matches!(observe_openfoam_boundary_patch(source, "inlet"), Err(OpenFoamBoundaryObservationError::UnterminatedBlockComment)));
    }
}
