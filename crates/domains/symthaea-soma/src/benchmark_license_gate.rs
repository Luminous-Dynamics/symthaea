// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fail-closed consistency checks for benchmark dataset rights manifests.
//!
//! This module validates a declared authorization record. It cannot establish
//! that a license is authentic, that a grant was actually issued, or that the
//! manifest's legal interpretation is correct. Those facts remain anchored in
//! private compliance records and counsel-reviewed source terms.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const BENCHMARK_LICENSE_MANIFEST_SCHEMA_V1: &str =
    "symthaea.benchmark-license-manifest.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkAccessStatus {
    Pending,
    Approved,
    Revoked,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkIntendedUse {
    NoncommercialResearch,
    AcademicPublication,
    CommercialEvaluation,
    CommercialDevelopment,
    Training,
    PublicMarketing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkOperation {
    Evaluation,
    Training,
    DerivativeMedia,
    Redistribution,
    Publication,
    CommercialUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkTransformation {
    None,
    AnnotationOnly,
    Trim,
    Resample,
    Normalization,
    TemporalShuffle,
    CounterfactualVisualSubstitution,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkLicenseManifestV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub dataset_manifest_digest: String,
    pub source_uri: String,
    pub rights_holder: String,
    pub license_id: String,
    pub license_version: String,
    pub access_status: BenchmarkAccessStatus,
    pub access_grant_ref: Option<String>,
    pub approved_at: Option<String>,
    pub expires_at: Option<String>,
    pub approved_intended_uses: Vec<BenchmarkIntendedUse>,
    pub permitted_operations: Vec<BenchmarkOperation>,
    pub intended_use: BenchmarkIntendedUse,
    pub commercial_use_authorized: bool,
    pub training_use_authorized: bool,
    pub derivative_media_authorized: bool,
    pub redistribution_authorized: bool,
    pub publication_authorized: bool,
    pub permitted_transformations: Vec<BenchmarkTransformation>,
    pub raw_media_stored_in_repo: bool,
    pub model_terms_ref: String,
    pub retention_policy_ref: String,
    pub deletion_policy_ref: String,
    pub approval_record_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkLicenseGateError {
    WrongSchema,
    InvalidIdentifier,
    InvalidDigest,
    InvalidSourceUri,
    InvalidUtcTimestamp,
    AccessNotApproved,
    MissingGrantReference,
    MissingApprovalReference,
    GrantNotYetApproved,
    GrantExpired,
    IntendedUseMismatch,
    IntendedUseNotApproved,
    OperationNotPermitted,
    CommercialUseNotAuthorized,
    TrainingNotAuthorized,
    DerivativeNotAuthorized,
    RedistributionNotAuthorized,
    PublicationNotAuthorized,
    TransformationNotPermitted,
    RawMediaRepositoryViolation,
    DuplicateApprovedUse,
    DuplicatePermittedOperation,
    DuplicateTransformation,
    InconsistentOperationFlags,
}

/// Validate the manifest for one exact intended use, operation, and optional
/// transformation. The caller must provide current UTC time in canonical
/// YYYY-MM-DDTHH:MM:SSZ form. Lexical comparison is valid only because both
/// timestamps are restricted to this same fixed-width UTC representation.
pub fn validate_benchmark_license_manifest(
    manifest: &BenchmarkLicenseManifestV1,
    now_utc: &str,
    requested_use: BenchmarkIntendedUse,
    required_operation: BenchmarkOperation,
    transformation: Option<BenchmarkTransformation>,
) -> Result<(), BenchmarkLicenseGateError> {
    if manifest.schema != BENCHMARK_LICENSE_MANIFEST_SCHEMA_V1 {
        return Err(BenchmarkLicenseGateError::WrongSchema);
    }
    validate_identifier(&manifest.dataset_id)?;
    validate_identifier(&manifest.dataset_version)?;
    validate_identifier(&manifest.license_id)?;
    validate_identifier(&manifest.license_version)?;
    validate_identifier(&manifest.model_terms_ref)?;
    validate_identifier(&manifest.retention_policy_ref)?;
    validate_identifier(&manifest.deletion_policy_ref)?;
    validate_digest(&manifest.dataset_manifest_digest)?;
    if manifest.rights_holder.trim().is_empty() || manifest.rights_holder.len() > 256 {
        return Err(BenchmarkLicenseGateError::InvalidIdentifier);
    }
    if !(manifest.source_uri.starts_with("https://")
        || manifest.source_uri.starts_with("http://")
        || manifest.source_uri.starts_with("urn:"))
    {
        return Err(BenchmarkLicenseGateError::InvalidSourceUri);
    }
    validate_utc_timestamp(now_utc)?;

    if manifest.raw_media_stored_in_repo {
        return Err(BenchmarkLicenseGateError::RawMediaRepositoryViolation);
    }
    if manifest.access_status != BenchmarkAccessStatus::Approved {
        return Err(BenchmarkLicenseGateError::AccessNotApproved);
    }

    let grant_ref = manifest
        .access_grant_ref
        .as_deref()
        .ok_or(BenchmarkLicenseGateError::MissingGrantReference)?;
    validate_identifier(grant_ref)?;
    let approval_ref = manifest
        .approval_record_ref
        .as_deref()
        .ok_or(BenchmarkLicenseGateError::MissingApprovalReference)?;
    validate_identifier(approval_ref)?;
    let approved_at = manifest
        .approved_at
        .as_deref()
        .ok_or(BenchmarkLicenseGateError::MissingApprovalReference)?;
    let expires_at = manifest
        .expires_at
        .as_deref()
        .ok_or(BenchmarkLicenseGateError::GrantExpired)?;
    validate_utc_timestamp(approved_at)?;
    validate_utc_timestamp(expires_at)?;
    if approved_at > now_utc {
        return Err(BenchmarkLicenseGateError::GrantNotYetApproved);
    }
    if expires_at <= now_utc {
        return Err(BenchmarkLicenseGateError::GrantExpired);
    }

    if requested_use != manifest.intended_use {
        return Err(BenchmarkLicenseGateError::IntendedUseMismatch);
    }
    ensure_unique(&manifest.approved_intended_uses)
        .map_err(|_| BenchmarkLicenseGateError::DuplicateApprovedUse)?;
    ensure_unique(&manifest.permitted_operations)
        .map_err(|_| BenchmarkLicenseGateError::DuplicatePermittedOperation)?;
    ensure_unique(&manifest.permitted_transformations)
        .map_err(|_| BenchmarkLicenseGateError::DuplicateTransformation)?;

    if !manifest.approved_intended_uses.contains(&requested_use) {
        return Err(BenchmarkLicenseGateError::IntendedUseNotApproved);
    }
    if !manifest.permitted_operations.contains(&required_operation) {
        return Err(BenchmarkLicenseGateError::OperationNotPermitted);
    }

    if is_commercial_use(requested_use) {
        if !manifest.commercial_use_authorized
            || !manifest.permitted_operations.contains(&BenchmarkOperation::CommercialUse)
        {
            return Err(BenchmarkLicenseGateError::CommercialUseNotAuthorized);
        }
    }
    if requested_use == BenchmarkIntendedUse::Training {
        if !manifest.training_use_authorized
            || !manifest.permitted_operations.contains(&BenchmarkOperation::Training)
        {
            return Err(BenchmarkLicenseGateError::TrainingNotAuthorized);
        }
    }
    if manifest.commercial_use_authorized
        != manifest.permitted_operations.contains(&BenchmarkOperation::CommercialUse)
        || manifest.training_use_authorized
            != manifest.permitted_operations.contains(&BenchmarkOperation::Training)
        || manifest.derivative_media_authorized
            != manifest.permitted_operations.contains(&BenchmarkOperation::DerivativeMedia)
        || manifest.redistribution_authorized
            != manifest.permitted_operations.contains(&BenchmarkOperation::Redistribution)
        || manifest.publication_authorized
            != manifest.permitted_operations.contains(&BenchmarkOperation::Publication)
    {
        return Err(BenchmarkLicenseGateError::InconsistentOperationFlags);
    }

    if required_operation == BenchmarkOperation::CommercialUse
        && !manifest.commercial_use_authorized
    {
        return Err(BenchmarkLicenseGateError::CommercialUseNotAuthorized);
    }
    if required_operation == BenchmarkOperation::Training && !manifest.training_use_authorized {
        return Err(BenchmarkLicenseGateError::TrainingNotAuthorized);
    }
    if required_operation == BenchmarkOperation::DerivativeMedia
        && !manifest.derivative_media_authorized
    {
        return Err(BenchmarkLicenseGateError::DerivativeNotAuthorized);
    }
    if required_operation == BenchmarkOperation::Redistribution
        && !manifest.redistribution_authorized
    {
        return Err(BenchmarkLicenseGateError::RedistributionNotAuthorized);
    }
    if required_operation == BenchmarkOperation::Publication && !manifest.publication_authorized {
        return Err(BenchmarkLicenseGateError::PublicationNotAuthorized);
    }

    if let Some(transform) = transformation {
        if !manifest.permitted_transformations.contains(&transform) {
            return Err(BenchmarkLicenseGateError::TransformationNotPermitted);
        }
        if matches!(
            transform,
            BenchmarkTransformation::Trim
                | BenchmarkTransformation::Resample
                | BenchmarkTransformation::Normalization
                | BenchmarkTransformation::TemporalShuffle
                | BenchmarkTransformation::CounterfactualVisualSubstitution
        ) && !manifest.derivative_media_authorized
        {
            return Err(BenchmarkLicenseGateError::DerivativeNotAuthorized);
        }
    }
    Ok(())
}

fn is_commercial_use(use_case: BenchmarkIntendedUse) -> bool {
    matches!(
        use_case,
        BenchmarkIntendedUse::CommercialEvaluation
            | BenchmarkIntendedUse::CommercialDevelopment
            | BenchmarkIntendedUse::PublicMarketing
    )
}

fn ensure_unique<T: Copy + Eq + std::hash::Hash>(values: &[T]) -> Result<(), ()> {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if !seen.insert(*value) {
            return Err(());
        }
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), BenchmarkLicenseGateError> {
    if value.is_empty()
        || value.len() > 160
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(BenchmarkLicenseGateError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), BenchmarkLicenseGateError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(BenchmarkLicenseGateError::InvalidDigest);
    }
    Ok(())
}

fn validate_utc_timestamp(value: &str) -> Result<(), BenchmarkLicenseGateError> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
        || bytes.iter().enumerate().any(|(index, byte)| {
            !matches!(index, 4 | 7 | 10 | 13 | 16 | 19) && !byte.is_ascii_digit()
        })
    {
        return Err(BenchmarkLicenseGateError::InvalidUtcTimestamp);
    }
    let year = parse_component(&value[0..4])?;
    let month = parse_component(&value[5..7])?;
    let day = parse_component(&value[8..10])?;
    let hour = parse_component(&value[11..13])?;
    let minute = parse_component(&value[14..16])?;
    let second = parse_component(&value[17..19])?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if day == 0
        || day > days_in_month
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(BenchmarkLicenseGateError::InvalidUtcTimestamp);
    }
    Ok(())
}

fn parse_component(value: &str) -> Result<u32, BenchmarkLicenseGateError> {
    value
        .parse::<u32>()
        .map_err(|_| BenchmarkLicenseGateError::InvalidUtcTimestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> BenchmarkLicenseManifestV1 {
        BenchmarkLicenseManifestV1 {
            schema: BENCHMARK_LICENSE_MANIFEST_SCHEMA_V1.to_owned(),
            dataset_id: "fixture-consented-av-corpus".to_owned(),
            dataset_version: "v1.0.0".to_owned(),
            dataset_manifest_digest: "a".repeat(64),
            source_uri: "urn:symthaea:fixture:consented-av-corpus:v1".to_owned(),
            rights_holder: "Fictional test rights holder".to_owned(),
            license_id: "fixture-commercial-research-license".to_owned(),
            license_version: "v1".to_owned(),
            access_status: BenchmarkAccessStatus::Approved,
            access_grant_ref: Some("fixture-only-grant-not-valid".to_owned()),
            approved_at: Some("2026-11-27T00:00:00Z".to_owned()),
            expires_at: Some("2027-11-27T00:00:00Z".to_owned()),
            approved_intended_uses: vec![
                BenchmarkIntendedUse::NoncommercialResearch,
                BenchmarkIntendedUse::CommercialEvaluation,
                BenchmarkIntendedUse::CommercialDevelopment,
            ],
            permitted_operations: vec![
                BenchmarkOperation::Evaluation,
                BenchmarkOperation::DerivativeMedia,
                BenchmarkOperation::Publication,
                BenchmarkOperation::CommercialUse,
            ],
            intended_use: BenchmarkIntendedUse::CommercialDevelopment,
            commercial_use_authorized: true,
            training_use_authorized: false,
            derivative_media_authorized: true,
            redistribution_authorized: false,
            publication_authorized: true,
            permitted_transformations: vec![
                BenchmarkTransformation::AnnotationOnly,
                BenchmarkTransformation::Trim,
                BenchmarkTransformation::TemporalShuffle,
                BenchmarkTransformation::CounterfactualVisualSubstitution,
            ],
            raw_media_stored_in_repo: false,
            model_terms_ref: "fixture-model-terms-v1".to_owned(),
            retention_policy_ref: "fixture-retention-policy-v1".to_owned(),
            deletion_policy_ref: "fixture-deletion-policy-v1".to_owned(),
            approval_record_ref: Some("fixture-only-approval-record".to_owned()),
        }
    }

    #[test]
    fn fictional_approved_manifest_passes_only_requested_scope() {
        let value = manifest();
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T00:00:00Z",
                BenchmarkIntendedUse::CommercialDevelopment,
                BenchmarkOperation::Evaluation,
                Some(BenchmarkTransformation::CounterfactualVisualSubstitution),
            ),
            Ok(())
        );
    }

    #[test]
    fn noncommercial_only_dataset_cannot_be_used_for_commercial_development() {
        let mut value = manifest();
        value.approved_intended_uses = vec![BenchmarkIntendedUse::NoncommercialResearch];
        value.permitted_operations = vec![BenchmarkOperation::Evaluation];
        value.intended_use = BenchmarkIntendedUse::NoncommercialResearch;
        value.commercial_use_authorized = false;
        value.derivative_media_authorized = false;
        value.publication_authorized = false;
        value.permitted_transformations = vec![BenchmarkTransformation::AnnotationOnly];
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T00:00:00Z",
                BenchmarkIntendedUse::CommercialDevelopment,
                BenchmarkOperation::Evaluation,
                None,
            ),
            Err(BenchmarkLicenseGateError::IntendedUseMismatch)
        );
    }

    #[test]
    fn evaluation_only_manifest_rejects_training() {
        let mut value = manifest();
        value.approved_intended_uses = vec![BenchmarkIntendedUse::NoncommercialResearch];
        value.permitted_operations = vec![BenchmarkOperation::Evaluation];
        value.intended_use = BenchmarkIntendedUse::NoncommercialResearch;
        value.commercial_use_authorized = false;
        value.derivative_media_authorized = false;
        value.publication_authorized = false;
        value.permitted_transformations = vec![BenchmarkTransformation::AnnotationOnly];
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T00:00:00Z",
                BenchmarkIntendedUse::NoncommercialResearch,
                BenchmarkOperation::Training,
                None,
            ),
            Err(BenchmarkLicenseGateError::OperationNotPermitted)
        );
    }

    #[test]
    fn unauthorized_counterfactual_transformation_is_rejected() {
        let mut value = manifest();
        value.derivative_media_authorized = false;
        value.permitted_operations.retain(|op| *op != BenchmarkOperation::DerivativeMedia);
        value.permitted_transformations = vec![BenchmarkTransformation::AnnotationOnly];
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T00:00:00Z",
                BenchmarkIntendedUse::CommercialDevelopment,
                BenchmarkOperation::Evaluation,
                Some(BenchmarkTransformation::CounterfactualVisualSubstitution),
            ),
            Err(BenchmarkLicenseGateError::TransformationNotPermitted)
        );
    }

    #[test]
    fn expired_grant_fails_closed() {
        let mut value = manifest();
        value.expires_at = Some("2026-11-27T00:00:00Z".to_owned());
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T00:00:00Z",
                BenchmarkIntendedUse::CommercialDevelopment,
                BenchmarkOperation::Evaluation,
                None,
            ),
            Err(BenchmarkLicenseGateError::GrantExpired)
        );
    }

    #[test]
    fn raw_media_in_public_repository_is_rejected() {
        let mut value = manifest();
        value.raw_media_stored_in_repo = true;
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T00:00:00Z",
                BenchmarkIntendedUse::CommercialDevelopment,
                BenchmarkOperation::Evaluation,
                None,
            ),
            Err(BenchmarkLicenseGateError::RawMediaRepositoryViolation)
        );
    }

    #[test]
    fn malformed_or_non_utc_clock_strings_are_rejected() {
        let value = manifest();
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T01:00:00+02:00",
                BenchmarkIntendedUse::CommercialDevelopment,
                BenchmarkOperation::Evaluation,
                None,
            ),
            Err(BenchmarkLicenseGateError::InvalidUtcTimestamp)
        );
    }

    #[test]
    fn commercial_operation_requires_explicit_commercial_authority() {
        let mut value = manifest();
        value.commercial_use_authorized = false;
        value.permitted_operations.retain(|op| *op != BenchmarkOperation::CommercialUse);
        assert_eq!(
            validate_benchmark_license_manifest(
                &value,
                "2026-11-28T00:00:00Z",
                BenchmarkIntendedUse::CommercialDevelopment,
                BenchmarkOperation::Evaluation,
                None,
            ),
            Err(BenchmarkLicenseGateError::CommercialUseNotAuthorized)
        );
    }
}
