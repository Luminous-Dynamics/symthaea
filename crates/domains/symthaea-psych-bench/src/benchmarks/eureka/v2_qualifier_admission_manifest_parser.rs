// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict parser for canonical EUREKA-002 V2 qualifier admission bytes.
//!
//! Signature verification must authenticate the exact bytes first; semantic
//! admission can then parse those same bytes through this fail-closed grammar.
//! No alternate field order, duplicate/unknown field, noncanonical integer, or
//! execution-authority escalation is accepted.

#![allow(dead_code)]

use super::v2_qualifier_admission_manifest::{
    V2QualifierAdmissionKind, V2QualifierAdmissionManifest,
    V2QualifierAdmissionManifestError, V2QualifierAdmissionRevisions,
    V2_QUALIFIER_ADMISSION_MANIFEST_SCHEMA, V2_REPOSITORY_IDENTITY,
};

const FIELD_COUNT: usize = 22;

pub(super) fn parse_canonical_admission_manifest(
    input: &[u8],
) -> Result<V2QualifierAdmissionManifest, V2QualifierAdmissionParseError> {
    let text = std::str::from_utf8(input)
        .map_err(|_| V2QualifierAdmissionParseError::InvalidUtf8)?;
    if !text.ends_with('\n') {
        return Err(V2QualifierAdmissionParseError::MissingTrailingNewline);
    }

    let body = &text[..text.len() - 1];
    let lines: Vec<&str> = body.split('\n').collect();
    if lines.len() != FIELD_COUNT {
        return Err(V2QualifierAdmissionParseError::WrongFieldCount);
    }

    let schema = field(lines[0], "manifest_schema_revision")?;
    if schema != V2_QUALIFIER_ADMISSION_MANIFEST_SCHEMA {
        return Err(V2QualifierAdmissionParseError::UnsupportedManifestSchema);
    }

    let kind = match field(lines[1], "admission_kind")? {
        "genesis" => V2QualifierAdmissionKind::Genesis,
        "rotation" => V2QualifierAdmissionKind::Rotation,
        _ => return Err(V2QualifierAdmissionParseError::InvalidAdmissionKind),
    };

    if field(lines[2], "repository")? != V2_REPOSITORY_IDENTITY {
        return Err(V2QualifierAdmissionParseError::WrongRepository);
    }

    let subject_head = field(lines[3], "subject_head")?;
    let subject_tree = field(lines[4], "subject_tree")?;
    let authority_schema_revision = field(lines[5], "authority_schema_revision")?;
    let authority_sequence = canonical_u64(field(lines[6], "authority_sequence")?)?;
    let predecessor_authority_commitment = optional_commitment(field(
        lines[7],
        "predecessor_authority_commitment",
    )?);
    let predecessor_currentness_commitment = optional_commitment(field(
        lines[8],
        "predecessor_currentness_commitment",
    )?);
    let authority_commitment = field(lines[9], "authority_commitment")?;
    let signer_policy_sequence = canonical_u64(field(lines[10], "signer_policy_sequence")?)?;
    let signer_policy_commitment = field(lines[11], "signer_policy_commitment")?;
    let receipt_schema_revision = field(lines[12], "receipt_schema_revision")?;
    let qualification_revision = field(lines[13], "qualification_revision")?;
    let command_contract_revision = field(lines[14], "command_contract_revision")?;
    let claim_scope = field(lines[15], "claim_scope")?;
    let workflow_sha256 = field(lines[16], "workflow_sha256")?;
    let command_contract_sha256 = field(lines[17], "command_contract_sha256")?;
    let environment_contract_revision = field(lines[18], "environment_contract_revision")?;
    let environment_commitment = field(lines[19], "environment_commitment")?;
    let evidence_capsule_sha256 = field(lines[20], "evidence_capsule_sha256")?;

    if field(lines[21], "execution_authority_granted")? != "false" {
        return Err(V2QualifierAdmissionParseError::ExecutionAuthorityRequested);
    }

    let revisions = V2QualifierAdmissionRevisions::explicit(
        authority_schema_revision,
        receipt_schema_revision,
        qualification_revision,
        command_contract_revision,
        claim_scope,
        environment_contract_revision,
    )
    .map_err(V2QualifierAdmissionParseError::Manifest)?;

    let manifest = V2QualifierAdmissionManifest::from_hex(
        kind,
        subject_head,
        subject_tree,
        revisions,
        authority_sequence,
        predecessor_authority_commitment,
        predecessor_currentness_commitment,
        authority_commitment,
        signer_policy_sequence,
        signer_policy_commitment,
        workflow_sha256,
        command_contract_sha256,
        environment_commitment,
        evidence_capsule_sha256,
    )
    .map_err(V2QualifierAdmissionParseError::Manifest)?;

    if manifest.canonical_bytes() != input {
        return Err(V2QualifierAdmissionParseError::NonCanonicalEncoding);
    }
    Ok(manifest)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierAdmissionParseError {
    InvalidUtf8,
    MissingTrailingNewline,
    WrongFieldCount,
    WrongField(&'static str),
    UnsupportedManifestSchema,
    InvalidAdmissionKind,
    WrongRepository,
    InvalidNumber,
    ExecutionAuthorityRequested,
    NonCanonicalEncoding,
    Manifest(V2QualifierAdmissionManifestError),
}

fn field<'a>(
    line: &'a str,
    expected_key: &'static str,
) -> Result<&'a str, V2QualifierAdmissionParseError> {
    let Some((key, value)) = line.split_once('=') else {
        return Err(V2QualifierAdmissionParseError::WrongField(expected_key));
    };
    if key != expected_key || value.contains('=') {
        return Err(V2QualifierAdmissionParseError::WrongField(expected_key));
    }
    Ok(value)
}

fn canonical_u64(value: &str) -> Result<u64, V2QualifierAdmissionParseError> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(V2QualifierAdmissionParseError::InvalidNumber);
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| V2QualifierAdmissionParseError::InvalidNumber)?;
    if parsed.to_string() != value {
        return Err(V2QualifierAdmissionParseError::InvalidNumber);
    }
    Ok(parsed)
}

fn optional_commitment(value: &str) -> Option<&str> {
    if value == "none" {
        None
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &[u8] = include_bytes!(
        "fixtures/eureka-v2-qualifier-admission-manifest-v1-genesis.env"
    );

    #[test]
    fn exact_golden_bytes_parse_and_round_trip() {
        let manifest = parse_canonical_admission_manifest(GOLDEN).unwrap();
        assert_eq!(manifest.canonical_bytes(), GOLDEN);
        assert_ne!(manifest.commitment(), [0_u8; 32]);
    }

    #[test]
    fn reordered_duplicate_unknown_or_extra_fields_fail_closed() {
        let text = std::str::from_utf8(GOLDEN).unwrap();
        let reordered = text.replacen(
            "admission_kind=genesis\nrepository=Luminous-Dynamics/symthaea\n",
            "repository=Luminous-Dynamics/symthaea\nadmission_kind=genesis\n",
            1,
        );
        assert!(matches!(
            parse_canonical_admission_manifest(reordered.as_bytes()),
            Err(V2QualifierAdmissionParseError::WrongField(_))
        ));

        let duplicated = text.replacen(
            "authority_sequence=1\n",
            "authority_sequence=1\nauthority_sequence=1\n",
            1,
        );
        assert_eq!(
            parse_canonical_admission_manifest(duplicated.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::WrongFieldCount
        );

        let unknown = text.replacen(
            "authority_sequence=1\n",
            "unknown_authority_sequence=1\n",
            1,
        );
        assert_eq!(
            parse_canonical_admission_manifest(unknown.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::WrongField("authority_sequence")
        );

        let extra = format!("{text}extra=true\n");
        assert_eq!(
            parse_canonical_admission_manifest(extra.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::WrongFieldCount
        );
    }

    #[test]
    fn missing_newline_and_noncanonical_numbers_fail_closed() {
        assert_eq!(
            parse_canonical_admission_manifest(&GOLDEN[..GOLDEN.len() - 1]).unwrap_err(),
            V2QualifierAdmissionParseError::MissingTrailingNewline
        );

        let text = std::str::from_utf8(GOLDEN).unwrap();
        let leading_zero = text.replacen("authority_sequence=1\n", "authority_sequence=01\n", 1);
        assert_eq!(
            parse_canonical_admission_manifest(leading_zero.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::InvalidNumber
        );
    }

    #[test]
    fn wrong_schema_repository_kind_and_execution_grant_fail_closed() {
        let text = std::str::from_utf8(GOLDEN).unwrap();
        let wrong_schema = text.replacen(
            V2_QUALIFIER_ADMISSION_MANIFEST_SCHEMA,
            "EUREKA.002.V2.QUALIFIER_ADMISSION_MANIFEST.v0",
            1,
        );
        assert_eq!(
            parse_canonical_admission_manifest(wrong_schema.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::UnsupportedManifestSchema
        );

        let wrong_repo = text.replacen(
            "repository=Luminous-Dynamics/symthaea",
            "repository=attacker/symthaea",
            1,
        );
        assert_eq!(
            parse_canonical_admission_manifest(wrong_repo.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::WrongRepository
        );

        let wrong_kind = text.replacen("admission_kind=genesis", "admission_kind=other", 1);
        assert_eq!(
            parse_canonical_admission_manifest(wrong_kind.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::InvalidAdmissionKind
        );

        let escalation = text.replacen(
            "execution_authority_granted=false",
            "execution_authority_granted=true",
            1,
        );
        assert_eq!(
            parse_canonical_admission_manifest(escalation.as_bytes()).unwrap_err(),
            V2QualifierAdmissionParseError::ExecutionAuthorityRequested
        );
    }

    #[test]
    fn signature_parser_source_contains_no_signing_or_execution_capability() {
        let production = include_str!("v2_qualifier_admission_manifest_parser.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "ssh-keygen -Y sign",
            "PRIVATE KEY",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "execution_authority_granted=true",
        ] {
            assert!(!production.contains(forbidden), "forbidden parser surface: {forbidden}");
        }
    }
}
