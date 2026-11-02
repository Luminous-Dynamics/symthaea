// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Source-bound identity joins for square connectome matrices.
//!
//! The intake layer supplies exact axis IDs, annotation IDs, dimensions and
//! source digests. This crate proves that the declared identity sets reconcile
//! and emits explicit permutations; it never guesses or coerces neuron IDs.

#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

pub const IDENTITY_SCHEMA_VERSION: &str = "CONNECTOME-AXIS-IDENTITY-V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatrixOrientation {
    RowsPresynapticColumnsPostsynaptic,
    RowsPostsynapticColumnsPresynaptic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactIdentity {
    pub dataset_id: String,
    pub release_id: String,
    pub taxon_id: String,
    pub life_stage_id: String,
    pub matrix_sha256: String,
    pub annotation_sha256: String,
    pub mapping_protocol_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixAxisInput {
    pub artifact: ArtifactIdentity,
    pub orientation: Option<MatrixOrientation>,
    pub matrix_rows: usize,
    pub matrix_columns: usize,
    pub row_ids: Vec<String>,
    pub column_ids: Vec<String>,
    pub annotation_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReasonCode {
    EmptyDatasetId,
    InvalidDatasetId,
    EmptyReleaseId,
    InvalidReleaseId,
    EmptyTaxonId,
    InvalidTaxonId,
    EmptyLifeStageId,
    InvalidLifeStageId,
    MalformedMatrixDigest,
    MalformedAnnotationDigest,
    MalformedMappingProtocolDigest,
    MissingOrientation,
    EmptyMatrix,
    MatrixNotSquare,
    MatrixPayloadRowCountMismatch,
    MatrixPayloadColumnCountMismatch,
    AnnotationPayloadLengthMismatch,
    RowAxisLengthMismatch,
    ColumnAxisLengthMismatch,
    EmptyRowId,
    InvalidRowId,
    DuplicateRowId,
    EmptyColumnId,
    InvalidColumnId,
    DuplicateColumnId,
    EmptyAnnotationId,
    InvalidAnnotationId,
    DuplicateAnnotationId,
    ColumnIdMissingFromRows,
    RowIdMissingFromColumns,
    AnnotationIdMissingFromRows,
    AnnotationIdMissingFromColumns,
    RowIdMissingFromAnnotations,
    ColumnIdMissingFromAnnotations,
    UnknownTargetId,
    InvalidTargetId,
    DuplicateTargetId,
    KeepRemoveConflict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFailure {
    reasons: Vec<ReasonCode>,
}

impl ValidationFailure {
    fn from_reasons(mut reasons: Vec<ReasonCode>) -> Self {
        reasons.sort_unstable();
        reasons.dedup();
        Self { reasons }
    }

    pub fn reasons(&self) -> &[ReasonCode] {
        &self.reasons
    }

    pub fn contains(&self, reason: ReasonCode) -> bool {
        self.reasons.binary_search(&reason).is_ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityJoin {
    schema_version: String,
    artifact: ArtifactIdentity,
    orientation: MatrixOrientation,
    canonical_ids: Vec<String>,
    row_to_canonical: Vec<usize>,
    column_to_canonical: Vec<usize>,
    annotation_to_canonical: Vec<usize>,
    canonical_to_row: Vec<usize>,
    canonical_to_column: Vec<usize>,
    canonical_to_annotation: Vec<usize>,
    /// Invariant under source-axis/table permutations with identical identities.
    canonical_identity_sha256: String,
    /// Binds the exact original order of rows, columns and annotation records.
    source_binding_sha256: String,
}

impl IdentityJoin {
    pub fn schema_version(&self) -> &str { &self.schema_version }
    pub fn artifact(&self) -> &ArtifactIdentity { &self.artifact }
    pub fn orientation(&self) -> MatrixOrientation { self.orientation }
    pub fn canonical_ids(&self) -> &[String] { &self.canonical_ids }
    pub fn row_to_canonical(&self) -> &[usize] { &self.row_to_canonical }
    pub fn column_to_canonical(&self) -> &[usize] { &self.column_to_canonical }
    pub fn annotation_to_canonical(&self) -> &[usize] { &self.annotation_to_canonical }
    pub fn canonical_to_row(&self) -> &[usize] { &self.canonical_to_row }
    pub fn canonical_to_column(&self) -> &[usize] { &self.canonical_to_column }
    pub fn canonical_to_annotation(&self) -> &[usize] { &self.canonical_to_annotation }
    pub fn canonical_identity_sha256(&self) -> &str { &self.canonical_identity_sha256 }
    pub fn source_binding_sha256(&self) -> &str { &self.source_binding_sha256 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTargets {
    ids: Vec<String>,
    canonical_indices: Vec<usize>,
}

impl ResolvedTargets {
    pub fn ids(&self) -> &[String] { &self.ids }
    pub fn canonical_indices(&self) -> &[usize] { &self.canonical_indices }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSelection {
    keep: Option<ResolvedTargets>,
    remove: ResolvedTargets,
}

impl ResolvedSelection {
    pub fn keep(&self) -> Option<&ResolvedTargets> { self.keep.as_ref() }
    pub fn remove(&self) -> &ResolvedTargets { &self.remove }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.chars().any(char::is_whitespace)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn push_identifier_error(
    value: &str,
    empty_reason: ReasonCode,
    invalid_reason: ReasonCode,
    reasons: &mut Vec<ReasonCode>,
) {
    if value.is_empty() {
        reasons.push(empty_reason);
    } else if !valid_identifier(value) {
        reasons.push(invalid_reason);
    }
}

fn unique_ids(
    ids: &[String],
    empty_reason: ReasonCode,
    invalid_reason: ReasonCode,
    duplicate_reason: ReasonCode,
    reasons: &mut Vec<ReasonCode>,
) -> BTreeSet<String> {
    let mut unique = BTreeSet::new();
    for id in ids {
        push_identifier_error(id, empty_reason, invalid_reason, reasons);
        if !unique.insert(id.clone()) {
            reasons.push(duplicate_reason);
        }
    }
    unique
}

fn push_len_prefixed(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn push_string(hasher: &mut Sha256, value: &str) {
    push_len_prefixed(hasher, value.as_bytes());
}

fn orientation_tag(orientation: MatrixOrientation) -> &'static str {
    match orientation {
        MatrixOrientation::RowsPresynapticColumnsPostsynaptic => "rows-pre-columns-post",
        MatrixOrientation::RowsPostsynapticColumnsPresynaptic => "rows-post-columns-pre",
    }
}

fn finish_digest(hasher: Sha256) -> String {
    let mut output = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn canonical_identity_digest(
    artifact: &ArtifactIdentity,
    orientation: MatrixOrientation,
    canonical_ids: &[String],
) -> String {
    let mut hasher = Sha256::new();
    push_string(&mut hasher, IDENTITY_SCHEMA_VERSION);
    push_string(&mut hasher, &artifact.dataset_id);
    push_string(&mut hasher, &artifact.release_id);
    push_string(&mut hasher, &artifact.taxon_id);
    push_string(&mut hasher, &artifact.life_stage_id);
    // Semantic identity excludes raw payload digests, which change when an
    // otherwise identical source export is reordered.
    push_string(&mut hasher, &artifact.mapping_protocol_sha256);
    push_string(&mut hasher, orientation_tag(orientation));
    for id in canonical_ids {
        push_string(&mut hasher, id);
    }
    finish_digest(hasher)
}

fn source_binding_digest(
    identity_digest: &str,
    matrix_sha256: &str,
    annotation_sha256: &str,
    matrix_rows: usize,
    matrix_columns: usize,
    row_ids: &[String],
    column_ids: &[String],
    annotation_ids: &[String],
) -> String {
    let mut hasher = Sha256::new();
    push_string(&mut hasher, "CONNECTOME-SOURCE-ORDER-BINDING-V1");
    push_string(&mut hasher, identity_digest);
    // Exact payload custody belongs here, not in the order-invariant semantic ID.
    push_string(&mut hasher, matrix_sha256);
    push_string(&mut hasher, annotation_sha256);
    hasher.update((matrix_rows as u64).to_le_bytes());
    hasher.update((matrix_columns as u64).to_le_bytes());
    for id in row_ids {
        push_string(&mut hasher, id);
    }
    for id in column_ids {
        push_string(&mut hasher, id);
    }
    for id in annotation_ids {
        push_string(&mut hasher, id);
    }
    finish_digest(hasher)
}

fn positions_for(source_ids: &[String], canonical_index: &BTreeMap<String, usize>) -> Vec<usize> {
    source_ids.iter().map(|id| canonical_index[id]).collect()
}

fn inverse_positions(source_to_canonical: &[usize], count: usize) -> Vec<usize> {
    let mut inverse = vec![usize::MAX; count];
    for (source_index, canonical_index) in source_to_canonical.iter().enumerate() {
        inverse[*canonical_index] = source_index;
    }
    inverse
}

/// Validates that matrix row IDs, column IDs and annotation IDs form one unique
/// neuron universe. Reordered inputs are accepted only with explicit bijections.
pub fn validate_identity_join(
    input: &MatrixAxisInput,
) -> Result<IdentityJoin, ValidationFailure> {
    let mut reasons = Vec::new();

    for (value, empty, invalid) in [
        (&input.artifact.dataset_id, ReasonCode::EmptyDatasetId, ReasonCode::InvalidDatasetId),
        (&input.artifact.release_id, ReasonCode::EmptyReleaseId, ReasonCode::InvalidReleaseId),
        (&input.artifact.taxon_id, ReasonCode::EmptyTaxonId, ReasonCode::InvalidTaxonId),
        (&input.artifact.life_stage_id, ReasonCode::EmptyLifeStageId, ReasonCode::InvalidLifeStageId),
    ] {
        push_identifier_error(value, empty, invalid, &mut reasons);
    }
    if !valid_sha256(&input.artifact.matrix_sha256) {
        reasons.push(ReasonCode::MalformedMatrixDigest);
    }
    if !valid_sha256(&input.artifact.annotation_sha256) {
        reasons.push(ReasonCode::MalformedAnnotationDigest);
    }
    if !valid_sha256(&input.artifact.mapping_protocol_sha256) {
        reasons.push(ReasonCode::MalformedMappingProtocolDigest);
    }

    let Some(orientation) = input.orientation else {
        reasons.push(ReasonCode::MissingOrientation);
        return Err(ValidationFailure::from_reasons(reasons));
    };
    if input.matrix_rows == 0 || input.matrix_columns == 0 {
        reasons.push(ReasonCode::EmptyMatrix);
    }
    if input.matrix_rows != input.matrix_columns {
        reasons.push(ReasonCode::MatrixNotSquare);
    }
    if input.row_ids.len() != input.matrix_rows {
        reasons.push(ReasonCode::RowAxisLengthMismatch);
    }
    if input.column_ids.len() != input.matrix_columns {
        reasons.push(ReasonCode::ColumnAxisLengthMismatch);
    }

    let row_set = unique_ids(
        &input.row_ids,
        ReasonCode::EmptyRowId,
        ReasonCode::InvalidRowId,
        ReasonCode::DuplicateRowId,
        &mut reasons,
    );
    let column_set = unique_ids(
        &input.column_ids,
        ReasonCode::EmptyColumnId,
        ReasonCode::InvalidColumnId,
        ReasonCode::DuplicateColumnId,
        &mut reasons,
    );
    let annotation_set = unique_ids(
        &input.annotation_ids,
        ReasonCode::EmptyAnnotationId,
        ReasonCode::InvalidAnnotationId,
        ReasonCode::DuplicateAnnotationId,
        &mut reasons,
    );

    if row_set.difference(&column_set).next().is_some() {
        reasons.push(ReasonCode::RowIdMissingFromColumns);
    }
    if column_set.difference(&row_set).next().is_some() {
        reasons.push(ReasonCode::ColumnIdMissingFromRows);
    }
    if annotation_set.difference(&row_set).next().is_some() {
        reasons.push(ReasonCode::AnnotationIdMissingFromRows);
    }
    if annotation_set.difference(&column_set).next().is_some() {
        reasons.push(ReasonCode::AnnotationIdMissingFromColumns);
    }
    if row_set.difference(&annotation_set).next().is_some() {
        reasons.push(ReasonCode::RowIdMissingFromAnnotations);
    }
    if column_set.difference(&annotation_set).next().is_some() {
        reasons.push(ReasonCode::ColumnIdMissingFromAnnotations);
    }

    if !reasons.is_empty() {
        return Err(ValidationFailure::from_reasons(reasons));
    }

    let canonical_ids: Vec<String> = row_set.into_iter().collect();
    let canonical_index: BTreeMap<String, usize> = canonical_ids
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();

    let row_to_canonical = positions_for(&input.row_ids, &canonical_index);
    let column_to_canonical = positions_for(&input.column_ids, &canonical_index);
    let annotation_to_canonical = positions_for(&input.annotation_ids, &canonical_index);
    let canonical_to_row = inverse_positions(&row_to_canonical, canonical_ids.len());
    let canonical_to_column = inverse_positions(&column_to_canonical, canonical_ids.len());
    let canonical_to_annotation = inverse_positions(&annotation_to_canonical, canonical_ids.len());

    let canonical_identity_sha256 =
        canonical_identity_digest(&input.artifact, orientation, &canonical_ids);
    let source_binding_sha256 = source_binding_digest(
        &canonical_identity_sha256,
        &input.artifact.matrix_sha256,
        &input.artifact.annotation_sha256,
        input.matrix_rows,
        input.matrix_columns,
        &input.row_ids,
        &input.column_ids,
        &input.annotation_ids,
    );

    Ok(IdentityJoin {
        schema_version: IDENTITY_SCHEMA_VERSION.to_string(),
        artifact: input.artifact.clone(),
        orientation,
        canonical_ids,
        row_to_canonical,
        column_to_canonical,
        annotation_to_canonical,
        canonical_to_row,
        canonical_to_column,
        canonical_to_annotation,
        canonical_identity_sha256,
        source_binding_sha256,
    })
}

/// Reorders the supplied source matrix into the join's canonical neuron order.
///
/// The matrix slice must use the same row/column order declared in the original
/// `MatrixAxisInput`. The join's immutable source-binding digest records that order.
pub fn canonicalize_matrix<T: Clone>(
    matrix: &[Vec<T>],
    join: &IdentityJoin,
) -> Result<Vec<Vec<T>>, ValidationFailure> {
    let count = join.canonical_ids.len();
    let mut reasons = Vec::new();
    if matrix.len() != count {
        reasons.push(ReasonCode::MatrixPayloadRowCountMismatch);
    }
    if matrix.iter().any(|row| row.len() != count) {
        reasons.push(ReasonCode::MatrixPayloadColumnCountMismatch);
    }
    if !reasons.is_empty() {
        return Err(ValidationFailure::from_reasons(reasons));
    }

    let mut canonical = Vec::with_capacity(count);
    for canonical_row in 0..count {
        let source_row = join.canonical_to_row[canonical_row];
        let mut row = Vec::with_capacity(count);
        for canonical_column in 0..count {
            let source_column = join.canonical_to_column[canonical_column];
            row.push(matrix[source_row][source_column].clone());
        }
        canonical.push(row);
    }
    Ok(canonical)
}

/// Reorders annotation or per-neuron parameter values into canonical neuron order.
pub fn canonicalize_annotation_values<T: Clone>(
    values: &[T],
    join: &IdentityJoin,
) -> Result<Vec<T>, ValidationFailure> {
    if values.len() != join.canonical_ids.len() {
        return Err(ValidationFailure::from_reasons(vec![
            ReasonCode::AnnotationPayloadLengthMismatch,
        ]));
    }
    Ok(join
        .canonical_to_annotation
        .iter()
        .map(|source_index| values[*source_index].clone())
        .collect())
}

/// Resolves stable neuron IDs into canonical indices; unknown/duplicate IDs reject.
pub fn resolve_targets(
    join: &IdentityJoin,
    target_ids: &[String],
) -> Result<ResolvedTargets, ValidationFailure> {
    let canonical_index: BTreeMap<&str, usize> = join
        .canonical_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();
    let mut reasons = Vec::new();
    let mut seen = BTreeSet::new();
    let mut resolved = Vec::with_capacity(target_ids.len());

    for id in target_ids {
        if !valid_identifier(id) {
            reasons.push(ReasonCode::InvalidTargetId);
            continue;
        }
        if !seen.insert(id.as_str()) {
            reasons.push(ReasonCode::DuplicateTargetId);
            continue;
        }
        match canonical_index.get(id.as_str()) {
            Some(index) => resolved.push((id.clone(), *index)),
            None => reasons.push(ReasonCode::UnknownTargetId),
        }
    }
    if !reasons.is_empty() {
        return Err(ValidationFailure::from_reasons(reasons));
    }

    resolved.sort_by_key(|(_, index)| *index);
    Ok(ResolvedTargets {
        ids: resolved.iter().map(|(id, _)| id.clone()).collect(),
        canonical_indices: resolved.iter().map(|(_, index)| *index).collect(),
    })
}

/// Resolves keep/remove requests by stable ID and rejects overlap.
pub fn resolve_selection(
    join: &IdentityJoin,
    keep_ids: Option<&[String]>,
    remove_ids: &[String],
) -> Result<ResolvedSelection, ValidationFailure> {
    let keep = keep_ids.map(|ids| resolve_targets(join, ids)).transpose()?;
    let remove = resolve_targets(join, remove_ids)?;

    if let Some(keep_targets) = &keep {
        let keep_set: BTreeSet<&str> = keep_targets.ids.iter().map(String::as_str).collect();
        if remove.ids.iter().any(|id| keep_set.contains(id.as_str())) {
            return Err(ValidationFailure::from_reasons(vec![ReasonCode::KeepRemoveConflict]));
        }
    }
    Ok(ResolvedSelection { keep, remove })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn artifact() -> ArtifactIdentity {
        ArtifactIdentity {
            dataset_id: "fly-vnc".to_string(),
            release_id: "manc-v1.2.1".to_string(),
            taxon_id: "NCBITaxon:7227".to_string(),
            life_stage_id: "adult".to_string(),
            matrix_sha256: digest('a'),
            annotation_sha256: digest('b'),
            mapping_protocol_sha256: digest('c'),
        }
    }

    fn input() -> MatrixAxisInput {
        MatrixAxisInput {
            artifact: artifact(),
            orientation: Some(MatrixOrientation::RowsPresynapticColumnsPostsynaptic),
            matrix_rows: 3,
            matrix_columns: 3,
            row_ids: vec!["100".to_string(), "200".to_string(), "300".to_string()],
            column_ids: vec!["100".to_string(), "200".to_string(), "300".to_string()],
            annotation_ids: vec!["100".to_string(), "200".to_string(), "300".to_string()],
        }
    }

    fn assert_reason(result: Result<IdentityJoin, ValidationFailure>, reason: ReasonCode) {
        let error = result.expect_err("invalid source identities must reject");
        assert!(error.contains(reason), "missing expected reason {reason:?}");
    }

    #[test]
    fn exact_axis_and_annotation_order_is_admitted() {
        let join = validate_identity_join(&input()).expect("exact identity join");
        assert_eq!(join.canonical_ids(), vec!["100", "200", "300"]);
        assert_eq!(join.row_to_canonical(), vec![0, 1, 2]);
        assert_eq!(join.column_to_canonical(), vec![0, 1, 2]);
        assert_eq!(join.annotation_to_canonical(), vec![0, 1, 2]);
    }

    #[test]
    fn independent_permutations_emit_explicit_bijections() {
        let mut source = input();
        source.row_ids = vec!["300".to_string(), "100".to_string(), "200".to_string()];
        source.column_ids = vec!["200".to_string(), "300".to_string(), "100".to_string()];
        source.annotation_ids = vec!["100".to_string(), "300".to_string(), "200".to_string()];
        let join = validate_identity_join(&source).expect("permuted identity join");
        assert_eq!(join.row_to_canonical(), vec![2, 0, 1]);
        assert_eq!(join.column_to_canonical(), vec![1, 2, 0]);
        assert_eq!(join.annotation_to_canonical(), vec![0, 2, 1]);
    }

    #[test]
    fn canonical_identity_digest_ignores_payload_order_but_binding_digest_does_not() {
        let first = validate_identity_join(&input()).expect("first join");
        let mut source = input();
        source.row_ids = vec!["200".to_string(), "300".to_string(), "100".to_string()];
        source.column_ids = vec!["300".to_string(), "100".to_string(), "200".to_string()];
        source.annotation_ids = vec!["200".to_string(), "100".to_string(), "300".to_string()];
        // Reordering a real export changes its exact payload digests.
        source.artifact.matrix_sha256 = digest('d');
        source.artifact.annotation_sha256 = digest('e');
        let second = validate_identity_join(&source).expect("permuted join");
        assert_eq!(first.canonical_identity_sha256(), second.canonical_identity_sha256());
        assert_ne!(first.source_binding_sha256(), second.source_binding_sha256());
    }
    #[test]
    fn leading_zero_ids_are_not_coerced() {
        let mut source = input();
        source.row_ids[0] = "00100".to_string();
        source.column_ids[0] = "00100".to_string();
        source.annotation_ids[0] = "00100".to_string();
        let join = validate_identity_join(&source).expect("leading-zero identity is valid");
        assert!(join.canonical_ids().contains(&"00100".to_string()));
        assert!(!join.canonical_ids().contains(&"100".to_string()));
    }

    #[test]
    fn missing_orientation_rejects() {
        let mut source = input();
        source.orientation = None;
        assert_reason(validate_identity_join(&source), ReasonCode::MissingOrientation);
    }

    #[test]
    fn non_square_matrix_rejects() {
        let mut source = input();
        source.matrix_rows = 2;
        assert_reason(validate_identity_join(&source), ReasonCode::MatrixNotSquare);
    }

    #[test]
    fn row_axis_length_must_match_dimensions() {
        let mut source = input();
        source.matrix_rows = 4;
        source.matrix_columns = 4;
        assert_reason(
            validate_identity_join(&source),
            ReasonCode::RowAxisLengthMismatch,
        );
    }

    #[test]
    fn duplicate_row_id_rejects() {
        let mut source = input();
        source.row_ids[1] = source.row_ids[0].clone();
        assert_reason(validate_identity_join(&source), ReasonCode::DuplicateRowId);
    }

    #[test]
    fn duplicate_column_id_rejects() {
        let mut source = input();
        source.column_ids[1] = source.column_ids[0].clone();
        assert_reason(
            validate_identity_join(&source),
            ReasonCode::DuplicateColumnId,
        );
    }

    #[test]
    fn duplicate_annotation_id_rejects() {
        let mut source = input();
        source.annotation_ids[1] = source.annotation_ids[0].clone();
        assert_reason(
            validate_identity_join(&source),
            ReasonCode::DuplicateAnnotationId,
        );
    }

    #[test]
    fn missing_annotation_mapping_rejects() {
        let mut source = input();
        source.annotation_ids[2] = "400".to_string();
        assert_reason(
            validate_identity_join(&source),
            ReasonCode::RowIdMissingFromAnnotations,
        );
    }

    #[test]
    fn mismatched_matrix_axes_reject() {
        let mut source = input();
        source.column_ids[2] = "400".to_string();
        assert_reason(
            validate_identity_join(&source),
            ReasonCode::ColumnIdMissingFromRows,
        );
    }

    #[test]
    fn malformed_source_digest_rejects() {
        let mut source = input();
        source.artifact.matrix_sha256 = "not-a-digest".to_string();
        assert_reason(
            validate_identity_join(&source),
            ReasonCode::MalformedMatrixDigest,
        );
    }

    #[test]
    fn canonicalize_matrix_applies_both_source_axis_permutations() {
        let mut source = input();
        source.row_ids = vec!["300".to_string(), "100".to_string(), "200".to_string()];
        source.column_ids = vec!["200".to_string(), "300".to_string(), "100".to_string()];
        source.annotation_ids = vec!["100".to_string(), "300".to_string(), "200".to_string()];
        let join = validate_identity_join(&source).expect("join");
        let matrix = vec![
            vec![1, 2, 0],
            vec![11, 12, 10],
            vec![21, 22, 20],
        ];
        let canonical = canonicalize_matrix(&matrix, &join).expect("canonical matrix");
        assert_eq!(canonical, vec![vec![10, 11, 12], vec![20, 21, 22], vec![0, 1, 2]]);
    }

    #[test]
    fn canonicalize_annotation_values_uses_annotation_permutation() {
        let mut source = input();
        source.annotation_ids = vec!["300".to_string(), "100".to_string(), "200".to_string()];
        let join = validate_identity_join(&source).expect("join");
        let source_values = vec!["C", "A", "B"];
        let canonical = canonicalize_annotation_values(&source_values, &join).expect("values");
        assert_eq!(canonical, vec!["A", "B", "C"]);
    }

    #[test]
    fn canonicalize_matrix_rejects_wrong_payload_shape() {
        let join = validate_identity_join(&input()).expect("join");
        let error = canonicalize_matrix(&[vec![1, 2], vec![3, 4]], &join)
            .expect_err("wrong shape must reject");
        assert!(error.contains(ReasonCode::MatrixPayloadRowCountMismatch));
        assert!(error.contains(ReasonCode::MatrixPayloadColumnCountMismatch));
    }

    #[test]
    fn canonicalize_annotation_values_rejects_wrong_length() {
        let join = validate_identity_join(&input()).expect("join");
        let error = canonicalize_annotation_values(&[1, 2], &join)
            .expect_err("wrong annotation length must reject");
        assert!(error.contains(ReasonCode::AnnotationPayloadLengthMismatch));
    }

    #[test]
    fn stable_target_resolution_is_order_independent() {
        let mut source = input();
        source.row_ids.reverse();
        source.column_ids.rotate_left(1);
        source.annotation_ids.reverse();
        let join = validate_identity_join(&source).expect("join");
        let resolved = resolve_targets(
            &join,
            &["300".to_string(), "100".to_string()],
        )
        .expect("stable target resolution");
        assert_eq!(resolved.ids(), vec!["100", "300"]);
        assert_eq!(resolved.canonical_indices(), vec![0, 2]);
    }

    #[test]
    fn invalid_target_identifier_rejects_without_normalization() {
        let join = validate_identity_join(&input()).expect("join");
        let error = resolve_targets(&join, &[" 100".to_string()])
            .expect_err("whitespace-mutated identity must reject");
        assert!(error.contains(ReasonCode::InvalidTargetId));
    }

    #[test]
    fn extra_annotation_identity_rejects() {
        let mut source = input();
        source.annotation_ids[2] = "400".to_string();
        let error = validate_identity_join(&source).expect_err("extra annotation must reject");
        assert!(error.contains(ReasonCode::AnnotationIdMissingFromRows));
    }

    #[test]
    fn matrix_orientation_is_part_of_canonical_identity() {
        let first = validate_identity_join(&input()).expect("first orientation");
        let mut source = input();
        source.orientation = Some(MatrixOrientation::RowsPostsynapticColumnsPresynaptic);
        let second = validate_identity_join(&source).expect("second orientation");
        assert_ne!(first.canonical_identity_sha256(), second.canonical_identity_sha256());
    }

    #[test]
    fn artifact_release_changes_canonical_identity() {
        let first = validate_identity_join(&input()).expect("first release");
        let mut source = input();
        source.artifact.release_id = "manc-v1.2.2".to_string();
        let second = validate_identity_join(&source).expect("second release");
        assert_ne!(first.canonical_identity_sha256(), second.canonical_identity_sha256());
    }

    #[test]
    fn unknown_keep_only_id_rejects() {
        let join = validate_identity_join(&input()).expect("join");
        let keep = vec!["999".to_string()];
        let error = resolve_selection(&join, Some(&keep), &[])
            .expect_err("unknown keep-only identity must reject");
        assert!(error.contains(ReasonCode::UnknownTargetId));
    }

    #[test]
    fn unknown_remove_id_rejects() {
        let join = validate_identity_join(&input()).expect("join");
        let remove = vec!["999".to_string()];
        let error = resolve_selection(&join, None, &remove)
            .expect_err("unknown remove identity must reject");
        assert!(error.contains(ReasonCode::UnknownTargetId));
    }

    #[test]
    fn unknown_target_id_rejects() {
        let join = validate_identity_join(&input()).expect("join");
        let error = resolve_targets(&join, &["999".to_string()])
            .expect_err("unknown stable ID must reject");
        assert!(error.contains(ReasonCode::UnknownTargetId));
    }

    #[test]
    fn duplicate_target_id_rejects() {
        let join = validate_identity_join(&input()).expect("join");
        let error = resolve_targets(
            &join,
            &["100".to_string(), "100".to_string()],
        )
        .expect_err("duplicate stable ID must reject");
        assert!(error.contains(ReasonCode::DuplicateTargetId));
    }

    #[test]
    fn keep_remove_overlap_rejects() {
        let join = validate_identity_join(&input()).expect("join");
        let keep = vec!["100".to_string(), "200".to_string()];
        let remove = vec!["200".to_string()];
        let error = resolve_selection(&join, Some(&keep), &remove)
            .expect_err("keep/remove overlap must reject");
        assert!(error.contains(ReasonCode::KeepRemoveConflict));
    }

    #[test]
    fn disjoint_keep_remove_sets_resolve_to_canonical_positions() {
        let join = validate_identity_join(&input()).expect("join");
        let keep = vec!["300".to_string(), "100".to_string()];
        let remove = vec!["200".to_string()];
        let result = resolve_selection(&join, Some(&keep), &remove).expect("disjoint selection");
        assert_eq!(result.keep().expect("keep set").canonical_indices(), vec![0, 2]);
        assert_eq!(result.remove().canonical_indices(), vec![1]);
    }

    #[test]
    fn whitespace_identity_is_not_normalized() {
        let mut source = input();
        source.row_ids[0] = " 100".to_string();
        assert_reason(validate_identity_join(&source), ReasonCode::InvalidRowId);
    }
}
