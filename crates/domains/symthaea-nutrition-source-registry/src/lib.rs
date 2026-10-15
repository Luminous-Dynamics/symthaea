// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! License-aware provenance gate for nutrition source records.
//!
//! This crate records declared rights and checks supplied metadata. It does not
//! authenticate a publisher, grant legal permission, validate a laboratory, decide
//! whether a diet is adequate, or calculate nutrient totals. A declared SHA-256 is
//! verified only when the corresponding artifact bytes are supplied.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use url::Url;

pub const NUTRITION_SOURCE_SCHEMA_VERSION: u16 = 1;

/// A permission decision is explicit. Unknown or conditional decisions never grant
/// permission merely because a source is publicly visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    AllowedByLicense,
    GrantedByWrittenPermission,
    Denied,
    RequiresWrittenPermission,
    Unknown,
}

impl PermissionDecision {
    fn is_granted(self) -> bool {
        matches!(
            self,
            Self::AllowedByLicense | Self::GrantedByWrittenPermission
        )
    }

    fn requires_evidence_reference(self) -> bool {
        // Both license grants and written permissions need auditable evidence.
        self.is_granted()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourcePermissionsV1 {
    pub use_for_assessment: PermissionDecision,
    pub cache: PermissionDecision,
    pub transform: PermissionDecision,
    pub redistribute: PermissionDecision,
    pub commercial_use: PermissionDecision,
    pub public_fixture: PermissionDecision,
}

/// Evidence references are scoped to a particular right. A reference documenting
/// assessment use must never be reused as evidence for commercial use or
/// redistribution. The referenced evidence itself still requires human review.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourcePermissionReferencesV1 {
    pub use_for_assessment: Option<String>,
    pub cache: Option<String>,
    pub transform: Option<String>,
    pub redistribute: Option<String>,
    pub commercial_use: Option<String>,
    pub public_fixture: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceRightV1 {
    Assessment,
    Cache,
    Transformation,
    Redistribution,
    CommercialUse,
    PublicFixture,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionEvidenceBasisV1 {
    LicenseTerms,
    WrittenPermission,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionEvidenceLifecycleV1 {
    Active,
    NeedsReview,
    Revoked,
    Withdrawn,
}

/// A reviewable, source-bound record referenced by one affirmative right decision.
/// These fields support auditability but do not authenticate the publisher or
/// determine legal sufficiency; reviewers must inspect the underlying evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionEvidenceV1 {
    pub evidence_id: String,
    pub source_id: String,
    pub right: SourceRightV1,
    pub basis: PermissionEvidenceBasisV1,
    pub evidence_uri: String,
    pub reviewed_on: String,
    pub valid_until: String,
    pub lifecycle: PermissionEvidenceLifecycleV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLifecycle {
    Active,
    NeedsReview,
    Revoked,
    Withdrawn,
}

/// Review dates are ISO-8601 calendar dates (YYYY-MM-DD); they are not claims
/// about the date an upstream publisher last changed a record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NutritionSourceManifestV1 {
    pub schema_version: u16,
    pub source_id: String,
    pub publisher: String,
    pub dataset_title: String,
    pub dataset_edition: String,
    pub source_uri: String,
    pub terms_uri: String,
    pub license_id: String,
    pub permission_references: SourcePermissionReferencesV1,
    /// Evidence records must resolve to the exact source and right they support.
    pub permission_evidence: Vec<PermissionEvidenceV1>,
    pub permissions: SourcePermissionsV1,
    /// SHA-256 of the exact upstream artifact bytes, lower-case hexadecimal.
    pub artifact_sha256: Option<String>,
    /// Date on which this source artifact/record was retrieved or acquired.
    pub retrieved_on: String,
    pub reviewed_on: String,
    pub review_valid_until: String,
    pub lifecycle: SourceLifecycle,
}

/// Declares how this caller plans to use the source-derived record. It is separate
/// from source license metadata so a source manifest cannot implicitly grant itself
/// the uses the caller requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceUseRequestV1 {
    pub retain_source_data: bool,
    pub transform_record: bool,
    pub commercial_use: bool,
    /// Publishing a fixture necessarily retains source-derived data.
    pub public_fixture: bool,
}

impl SourceUseRequestV1 {
    fn retains_source_data(&self) -> bool {
        self.retain_source_data || self.public_fixture
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NutrientUnit {
    Micrograms,
    Milligrams,
    Grams,
    Kilocalories,
    Kilojoules,
    Millilitres,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    LaboratoryAnalysis,
    NationalFoodCompositionTable,
    PublishedLiterature,
    FoodLabel,
    ReviewedEstimate,
    SyntheticFixture,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoodMatchQuality {
    ExactReviewedMatch,
    RegionalMatch,
    ApproximateMatch,
    Unresolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoodGeographicMatch {
    SouthAfricanLocal,
    SouthernAfricanRegional,
    GlobalFallback,
    Unknown,
}

/// One explicit preparation/conversion step between food states. References should
/// resolve to reviewed method and evidence records in the source manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NutritionTransformationStepV1 {
    pub transformation_id: String,
    pub from_food_state_id: String,
    pub to_food_state_id: String,
    pub method_ref: String,
    pub evidence_ref: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoodIdentityConfidence {
    Reviewed,
    Probable,
    Unknown,
}

/// A source-attributed food fact; numeric values use an explicit integer scale.
/// A missing value is None, never an implicit zero. This record contains no
/// individual health or household identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NutritionFoodFactV1 {
    pub schema_version: u16,
    pub source_id: String,
    pub source_record_id: String,
    pub source_record_citation: String,
    pub food_id: String,
    pub food_description: String,
    pub food_taxon_ref: Option<String>,
    pub cultivar_id: Option<String>,
    pub identity_confidence: FoodIdentityConfidence,
    pub food_state_id: String,
    pub match_quality: FoodMatchQuality,
    pub geographic_match: FoodGeographicMatch,
    pub evidence_class: EvidenceClass,
    pub nutrient_id: String,
    pub nutrient_definition_ref: String,
    pub unit: NutrientUnit,
    pub value_scaled: Option<u64>,
    pub value_scale: u64,
    pub basis_grams: u64,
    pub quality_notes: Vec<String>,
    pub transformation_applied: bool,
    pub transformation_lineage: Vec<NutritionTransformationStepV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Eligibility {
    Eligible,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataQualityEligibility {
    ExactMatch,
    QualifiedFallback,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactIntegrity {
    Verified,
    NoDigestClaimed,
    Unverified,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NutritionAssessability {
    Assessable,
    QualifiedFallback,
    NotAssessable,
    SyntheticFixtureOnly,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueCategory {
    Source,
    License,
    DataQuality,
    Integrity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryIssueCode {
    UnsupportedSourceSchema,
    MissingSourceId,
    MissingPublisher,
    MissingDatasetTitle,
    MissingDatasetEdition,
    InvalidSourceUri,
    InvalidTermsUri,
    MissingLicenseId,
    InvalidEvaluationDate,
    InvalidReviewedOn,
    InvalidRetrievedOn,
    SourceRetrievedInFuture,
    InvalidReviewValidUntil,
    ReviewWindowInverted,
    ReviewNotYetEffective,
    ReviewExpired,
    SourceNeedsReview,
    SourceRevoked,
    SourceWithdrawn,
    AssessmentPermissionReferenceMissing,
    CachePermissionReferenceMissing,
    TransformationPermissionReferenceMissing,
    CommercialUsePermissionReferenceMissing,
    PublicFixturePermissionReferenceMissing,
    RedistributionPermissionReferenceMissing,
    PermissionEvidenceNotFound,
    PermissionEvidenceAmbiguous,
    PermissionEvidenceSourceMismatch,
    PermissionEvidenceRightMismatch,
    PermissionEvidenceBasisMismatch,
    PermissionEvidenceUriInvalid,
    PermissionEvidenceReviewedDateInvalid,
    PermissionEvidenceValidityDateInvalid,
    PermissionEvidenceWindowInverted,
    PermissionEvidenceNotYetEffective,
    PermissionEvidenceExpired,
    PermissionEvidenceNeedsReview,
    PermissionEvidenceRevoked,
    PermissionEvidenceWithdrawn,
    AssessmentUseNotPermitted,
    CacheNotPermitted,
    TransformationNotPermitted,
    CommercialUseNotPermitted,
    PublicFixtureNotPermitted,
    RedistributionNotPermitted,
    UnsupportedFactSchema,
    FactSourceMismatch,
    MissingSourceRecordId,
    MissingSourceCitation,
    MissingFoodId,
    MissingFoodDescription,
    EmptyFoodTaxonReference,
    EmptyCultivarId,
    UnknownIdentityConfidence,
    MissingFoodState,
    MissingNutrientId,
    MissingNutrientDefinition,
    ZeroValueScale,
    ZeroBasisMass,
    UnresolvedFoodMatch,
    UnknownGeographicMatch,
    MissingTransformationLineage,
    UnexpectedTransformationLineage,
    MissingTransformationId,
    MissingTransformationFromState,
    MissingTransformationToState,
    MissingTransformationMethod,
    MissingTransformationEvidence,
    TransformationChainMismatch,
    TransformationOutputMismatch,
    MissingArtifactDigest,
    InvalidArtifactDigest,
    ArtifactBytesRequired,
    ArtifactDigestUnverified,
    ArtifactDigestMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryIssueV1 {
    pub category: IssueCategory,
    pub code: RegistryIssueCode,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NutritionRegistryAssessmentV1 {
    pub source_eligibility: Eligibility,
    pub license_eligibility: Eligibility,
    pub data_quality_eligibility: DataQualityEligibility,
    pub artifact_integrity: ArtifactIntegrity,
    pub nutrition_assessability: NutritionAssessability,
    /// Stable ordering follows the evaluator's fixed validation sequence.
    pub issues: Vec<RegistryIssueV1>,
}

/// Returns a lowercase SHA-256 digest of exactly the supplied bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Evaluate one fact against a reviewed manifest for the supplied date.
///
/// If an artifact digest is declared, the exact bytes must be supplied to verify it.
/// If data are retained, an explicit digest and matching bytes are mandatory.
/// An absent digest is reported as "no digest claimed", not as proof of integrity.
pub fn assess_nutrition_source(
    manifest: &NutritionSourceManifestV1,
    fact: &NutritionFoodFactV1,
    evaluation_date: &str,
    use_request: &SourceUseRequestV1,
    artifact_bytes: Option<&[u8]>,
) -> NutritionRegistryAssessmentV1 {
    let mut issues = Vec::new();

    validate_source_metadata(manifest, evaluation_date, &mut issues);
    validate_permissions(manifest, fact, evaluation_date, use_request, &mut issues);
    validate_fact(manifest, fact, &mut issues);
    let artifact_integrity =
        validate_artifact_digest(manifest, use_request, artifact_bytes, &mut issues);

    let source_eligibility = if issues.iter().any(|issue| issue.category == IssueCategory::Source) {
        Eligibility::Blocked
    } else {
        Eligibility::Eligible
    };
    let license_eligibility = if issues
        .iter()
        .any(|issue| issue.category == IssueCategory::License)
    {
        Eligibility::Blocked
    } else {
        Eligibility::Eligible
    };
    let data_quality_eligibility = if issues
        .iter()
        .any(|issue| issue.category == IssueCategory::DataQuality)
    {
        DataQualityEligibility::Blocked
    } else if is_qualified_fallback(fact) {
        DataQualityEligibility::QualifiedFallback
    } else {
        DataQualityEligibility::ExactMatch
    };

    let any_blocker = !issues.is_empty();
    let nutrition_assessability = if any_blocker {
        NutritionAssessability::Blocked
    } else if fact.value_scaled.is_none() {
        NutritionAssessability::NotAssessable
    } else if fact.evidence_class == EvidenceClass::SyntheticFixture {
        NutritionAssessability::SyntheticFixtureOnly
    } else if is_qualified_fallback(fact) {
        NutritionAssessability::QualifiedFallback
    } else {
        NutritionAssessability::Assessable
    };

    NutritionRegistryAssessmentV1 {
        source_eligibility,
        license_eligibility,
        data_quality_eligibility,
        artifact_integrity,
        nutrition_assessability,
        issues,
    }
}

fn validate_source_metadata(
    manifest: &NutritionSourceManifestV1,
    evaluation_date: &str,
    issues: &mut Vec<RegistryIssueV1>,
) {
    if manifest.schema_version != NUTRITION_SOURCE_SCHEMA_VERSION {
        push_issue(issues, IssueCategory::Source, RegistryIssueCode::UnsupportedSourceSchema);
    }
    required_source_field(&manifest.source_id, RegistryIssueCode::MissingSourceId, issues);
    required_source_field(&manifest.publisher, RegistryIssueCode::MissingPublisher, issues);
    required_source_field(&manifest.dataset_title, RegistryIssueCode::MissingDatasetTitle, issues);
    required_source_field(
        &manifest.dataset_edition,
        RegistryIssueCode::MissingDatasetEdition,
        issues,
    );
    required_source_field(&manifest.license_id, RegistryIssueCode::MissingLicenseId, issues);
    if !valid_http_uri(&manifest.source_uri) {
        push_issue(issues, IssueCategory::Source, RegistryIssueCode::InvalidSourceUri);
    }
    if !valid_http_uri(&manifest.terms_uri) {
        push_issue(issues, IssueCategory::Source, RegistryIssueCode::InvalidTermsUri);
    }

    let retrieved = parse_date(&manifest.retrieved_on);
    if retrieved.is_none() {
        push_issue(issues, IssueCategory::Source, RegistryIssueCode::InvalidRetrievedOn);
    }

    match parse_date(evaluation_date) {
        None => push_issue(issues, IssueCategory::Source, RegistryIssueCode::InvalidEvaluationDate),
        Some(as_of) => {
            if retrieved.is_some_and(|retrieved_on| retrieved_on > as_of) {
                push_issue(
                    issues,
                    IssueCategory::Source,
                    RegistryIssueCode::SourceRetrievedInFuture,
                );
            }
            let reviewed = parse_date(&manifest.reviewed_on);
            let until = parse_date(&manifest.review_valid_until);
            if reviewed.is_none() {
                push_issue(issues, IssueCategory::Source, RegistryIssueCode::InvalidReviewedOn);
            }
            if until.is_none() {
                push_issue(
                    issues,
                    IssueCategory::Source,
                    RegistryIssueCode::InvalidReviewValidUntil,
                );
            }
            if let (Some(reviewed), Some(until)) = (reviewed, until) {
                if until < reviewed {
                    push_issue(
                        issues,
                        IssueCategory::Source,
                        RegistryIssueCode::ReviewWindowInverted,
                    );
                } else if as_of < reviewed {
                    push_issue(
                        issues,
                        IssueCategory::Source,
                        RegistryIssueCode::ReviewNotYetEffective,
                    );
                } else if as_of > until {
                    push_issue(issues, IssueCategory::Source, RegistryIssueCode::ReviewExpired);
                }
            }
        }
    }

    match manifest.lifecycle {
        SourceLifecycle::Active => {}
        SourceLifecycle::NeedsReview => {
            push_issue(issues, IssueCategory::Source, RegistryIssueCode::SourceNeedsReview)
        }
        SourceLifecycle::Revoked => {
            push_issue(issues, IssueCategory::Source, RegistryIssueCode::SourceRevoked)
        }
        SourceLifecycle::Withdrawn => {
            push_issue(issues, IssueCategory::Source, RegistryIssueCode::SourceWithdrawn)
        }
    }
}

fn validate_permission_reference(
    manifest: &NutritionSourceManifestV1,
    decision: PermissionDecision,
    right: SourceRightV1,
    reference: Option<&str>,
    missing_code: RegistryIssueCode,
    evaluation_date: &str,
    issues: &mut Vec<RegistryIssueV1>,
) {
    if !decision.requires_evidence_reference() {
        return;
    }
    let Some(reference) = reference.map(str::trim).filter(|value| !value.is_empty()) else {
        push_issue(issues, IssueCategory::License, missing_code);
        return;
    };

    let matches: Vec<_> = manifest.permission_evidence.iter()
        .filter(|evidence| evidence.evidence_id == reference)
        .collect();
    let evidence = match matches.as_slice() {
        [] => {
            push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceNotFound);
            return;
        }
        [evidence] => *evidence,
        _ => {
            push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceAmbiguous);
            return;
        }
    };

    if evidence.source_id != manifest.source_id {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceSourceMismatch);
    }
    if evidence.right != right {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceRightMismatch);
    }
    let expected_basis = match decision {
        PermissionDecision::AllowedByLicense => PermissionEvidenceBasisV1::LicenseTerms,
        PermissionDecision::GrantedByWrittenPermission => PermissionEvidenceBasisV1::WrittenPermission,
        PermissionDecision::Denied | PermissionDecision::RequiresWrittenPermission | PermissionDecision::Unknown => {
            return;
        }
    };
    if evidence.basis != expected_basis {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceBasisMismatch);
    }
    if !valid_http_uri(&evidence.evidence_uri) {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceUriInvalid);
    }

    let reviewed = parse_date(&evidence.reviewed_on);
    let until = parse_date(&evidence.valid_until);
    let as_of = parse_date(evaluation_date);
    if reviewed.is_none() {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceReviewedDateInvalid);
    }
    if until.is_none() {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceValidityDateInvalid);
    }
    if let (Some(reviewed), Some(until)) = (reviewed, until) {
        if until < reviewed {
            push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceWindowInverted);
        } else if let Some(as_of) = as_of {
            if as_of < reviewed {
                push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceNotYetEffective);
            } else if as_of > until {
                push_issue(issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceExpired);
            }
        }
    }
    match evidence.lifecycle {
        PermissionEvidenceLifecycleV1::Active => {}
        PermissionEvidenceLifecycleV1::NeedsReview => push_issue(
            issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceNeedsReview
        ),
        PermissionEvidenceLifecycleV1::Revoked => push_issue(
            issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceRevoked
        ),
        PermissionEvidenceLifecycleV1::Withdrawn => push_issue(
            issues, IssueCategory::License, RegistryIssueCode::PermissionEvidenceWithdrawn
        ),
    }
}

fn validate_permissions(
    manifest: &NutritionSourceManifestV1,
    fact: &NutritionFoodFactV1,
    evaluation_date: &str,
    use_request: &SourceUseRequestV1,
    issues: &mut Vec<RegistryIssueV1>,
) {
    let p = &manifest.permissions;
    let r = &manifest.permission_references;

    validate_permission_reference(
        manifest, p.use_for_assessment, SourceRightV1::Assessment,
        r.use_for_assessment.as_deref(),
        RegistryIssueCode::AssessmentPermissionReferenceMissing, evaluation_date, issues,
    );
    if use_request.retains_source_data() {
        validate_permission_reference(
            manifest, p.cache, SourceRightV1::Cache, r.cache.as_deref(),
            RegistryIssueCode::CachePermissionReferenceMissing, evaluation_date, issues,
        );
    }
    if fact.transformation_applied || use_request.transform_record {
        validate_permission_reference(
            manifest, p.transform, SourceRightV1::Transformation, r.transform.as_deref(),
            RegistryIssueCode::TransformationPermissionReferenceMissing, evaluation_date, issues,
        );
    }
    if use_request.commercial_use {
        validate_permission_reference(
            manifest, p.commercial_use, SourceRightV1::CommercialUse, r.commercial_use.as_deref(),
            RegistryIssueCode::CommercialUsePermissionReferenceMissing, evaluation_date, issues,
        );
    }
    if use_request.public_fixture {
        validate_permission_reference(
            manifest, p.public_fixture, SourceRightV1::PublicFixture, r.public_fixture.as_deref(),
            RegistryIssueCode::PublicFixturePermissionReferenceMissing, evaluation_date, issues,
        );
        validate_permission_reference(
            manifest, p.redistribute, SourceRightV1::Redistribution, r.redistribute.as_deref(),
            RegistryIssueCode::RedistributionPermissionReferenceMissing, evaluation_date, issues,
        );
    }
    if !p.use_for_assessment.is_granted() {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::AssessmentUseNotPermitted);
    }
    if use_request.retains_source_data() && !p.cache.is_granted() {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::CacheNotPermitted);
    }
    if (fact.transformation_applied || use_request.transform_record)
        && !p.transform.is_granted() {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::TransformationNotPermitted);
    }
    if use_request.commercial_use && !p.commercial_use.is_granted() {
        push_issue(issues, IssueCategory::License, RegistryIssueCode::CommercialUseNotPermitted);
    }
    if use_request.public_fixture {
        if !p.public_fixture.is_granted() {
            push_issue(
                issues,
                IssueCategory::License,
                RegistryIssueCode::PublicFixtureNotPermitted,
            );
        }
        if !p.redistribute.is_granted() {
            push_issue(
                issues,
                IssueCategory::License,
                RegistryIssueCode::RedistributionNotPermitted,
            );
        }
    }
}

fn validate_fact(
    manifest: &NutritionSourceManifestV1,
    fact: &NutritionFoodFactV1,
    issues: &mut Vec<RegistryIssueV1>,
) {
    if fact.schema_version != NUTRITION_SOURCE_SCHEMA_VERSION {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::UnsupportedFactSchema);
    }
    if fact.source_id != manifest.source_id {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::FactSourceMismatch);
    }
    required_fact_field(&fact.source_record_id, RegistryIssueCode::MissingSourceRecordId, issues);
    required_fact_field(
        &fact.source_record_citation,
        RegistryIssueCode::MissingSourceCitation,
        issues,
    );
    required_fact_field(&fact.food_id, RegistryIssueCode::MissingFoodId, issues);
    required_fact_field(&fact.food_description, RegistryIssueCode::MissingFoodDescription, issues);
    if fact.food_taxon_ref.as_ref().is_some_and(|value| value.trim().is_empty()) {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::EmptyFoodTaxonReference);
    }
    if fact.cultivar_id.as_ref().is_some_and(|value| value.trim().is_empty()) {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::EmptyCultivarId);
    }
    if fact.identity_confidence == FoodIdentityConfidence::Unknown {
        push_issue(
            issues,
            IssueCategory::DataQuality,
            RegistryIssueCode::UnknownIdentityConfidence,
        );
    }
    required_fact_field(&fact.food_state_id, RegistryIssueCode::MissingFoodState, issues);
    required_fact_field(&fact.nutrient_id, RegistryIssueCode::MissingNutrientId, issues);
    required_fact_field(
        &fact.nutrient_definition_ref,
        RegistryIssueCode::MissingNutrientDefinition,
        issues,
    );
    if fact.value_scale == 0 {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::ZeroValueScale);
    }
    if fact.basis_grams == 0 {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::ZeroBasisMass);
    }
    if fact.match_quality == FoodMatchQuality::Unresolved {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::UnresolvedFoodMatch);
    }
    if fact.geographic_match == FoodGeographicMatch::Unknown {
        push_issue(issues, IssueCategory::DataQuality, RegistryIssueCode::UnknownGeographicMatch);
    }

    if fact.transformation_applied && fact.transformation_lineage.is_empty() {
        push_issue(
            issues,
            IssueCategory::DataQuality,
            RegistryIssueCode::MissingTransformationLineage,
        );
    }
    if !fact.transformation_applied && !fact.transformation_lineage.is_empty() {
        push_issue(
            issues,
            IssueCategory::DataQuality,
            RegistryIssueCode::UnexpectedTransformationLineage,
        );
    }
    let mut previous_output: Option<&str> = None;
    for step in &fact.transformation_lineage {
        if step.transformation_id.trim().is_empty() {
            push_issue(
                issues,
                IssueCategory::DataQuality,
                RegistryIssueCode::MissingTransformationId,
            );
        }
        if step.from_food_state_id.trim().is_empty() {
            push_issue(
                issues,
                IssueCategory::DataQuality,
                RegistryIssueCode::MissingTransformationFromState,
            );
        }
        if step.to_food_state_id.trim().is_empty() {
            push_issue(
                issues,
                IssueCategory::DataQuality,
                RegistryIssueCode::MissingTransformationToState,
            );
        }
        if step.method_ref.trim().is_empty() {
            push_issue(
                issues,
                IssueCategory::DataQuality,
                RegistryIssueCode::MissingTransformationMethod,
            );
        }
        if step.evidence_ref.trim().is_empty() {
            push_issue(
                issues,
                IssueCategory::DataQuality,
                RegistryIssueCode::MissingTransformationEvidence,
            );
        }
        if let Some(output) = previous_output {
            if !output.is_empty()
                && !step.from_food_state_id.trim().is_empty()
                && output != step.from_food_state_id.as_str()
            {
                push_issue(
                    issues,
                    IssueCategory::DataQuality,
                    RegistryIssueCode::TransformationChainMismatch,
                );
            }
        }
        previous_output = Some(&step.to_food_state_id);
    }
    if let Some(output) = previous_output {
        if !output.trim().is_empty() && output != fact.food_state_id.as_str() {
            push_issue(
                issues,
                IssueCategory::DataQuality,
                RegistryIssueCode::TransformationOutputMismatch,
            );
        }
    }
}

fn validate_artifact_digest(
    manifest: &NutritionSourceManifestV1,
    use_request: &SourceUseRequestV1,
    artifact_bytes: Option<&[u8]>,
    issues: &mut Vec<RegistryIssueV1>,
) -> ArtifactIntegrity {
    let digest = manifest.artifact_sha256.as_deref();
    if use_request.retains_source_data() && digest.is_none() {
        push_issue(issues, IssueCategory::Integrity, RegistryIssueCode::MissingArtifactDigest);
    }
    if let Some(declared) = digest {
        if !is_canonical_sha256(declared) {
            push_issue(issues, IssueCategory::Integrity, RegistryIssueCode::InvalidArtifactDigest);
            return ArtifactIntegrity::Invalid;
        }
        let Some(bytes) = artifact_bytes else {
            push_issue(
                issues,
                IssueCategory::Integrity,
                RegistryIssueCode::ArtifactDigestUnverified,
            );
            if use_request.retains_source_data() {
                push_issue(
                    issues,
                    IssueCategory::Integrity,
                    RegistryIssueCode::ArtifactBytesRequired,
                );
            }
            return ArtifactIntegrity::Unverified;
        };
        if sha256_hex(bytes) != declared {
            push_issue(issues, IssueCategory::Integrity, RegistryIssueCode::ArtifactDigestMismatch);
            return ArtifactIntegrity::Invalid;
        }
        return ArtifactIntegrity::Verified;
    }

    if artifact_bytes.is_some() && use_request.retains_source_data() {
        return ArtifactIntegrity::Invalid;
    }
    if use_request.retains_source_data() {
        ArtifactIntegrity::Invalid
    } else {
        ArtifactIntegrity::NoDigestClaimed
    }
}

fn is_qualified_fallback(fact: &NutritionFoodFactV1) -> bool {
    fact.match_quality != FoodMatchQuality::ExactReviewedMatch
        || fact.geographic_match != FoodGeographicMatch::SouthAfricanLocal
        || fact.identity_confidence == FoodIdentityConfidence::Probable
}

fn valid_http_uri(raw: &str) -> bool {
    Url::parse(raw)
        .map(|uri| matches!(uri.scheme(), "http" | "https") && uri.host_str().is_some())
        .unwrap_or(false)
}

fn parse_date(raw: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(raw, "%Y-%m-%d")
        .ok()
        .filter(|date| date.format("%Y-%m-%d").to_string() == raw)
}

fn is_canonical_sha256(raw: &str) -> bool {
    raw.len() == 64
        && raw
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_source_field(
    value: &str,
    code: RegistryIssueCode,
    issues: &mut Vec<RegistryIssueV1>,
) {
    if value.trim().is_empty() {
        push_issue(issues, IssueCategory::Source, code);
    }
}

fn required_fact_field(
    value: &str,
    code: RegistryIssueCode,
    issues: &mut Vec<RegistryIssueV1>,
) {
    if value.trim().is_empty() {
        push_issue(issues, IssueCategory::DataQuality, code);
    }
}

fn push_issue(
    issues: &mut Vec<RegistryIssueV1>,
    category: IssueCategory,
    code: RegistryIssueCode,
) {
    issues.push(RegistryIssueV1 { category, code });
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVALUATION_DATE: &str = "2026-10-15";
    const ARTIFACT: &[u8] = b"synthetic nutrition source artifact v1";

    fn allowed_permissions() -> SourcePermissionsV1 {
        SourcePermissionsV1 {
            use_for_assessment: PermissionDecision::AllowedByLicense,
            cache: PermissionDecision::AllowedByLicense,
            transform: PermissionDecision::AllowedByLicense,
            redistribute: PermissionDecision::AllowedByLicense,
            commercial_use: PermissionDecision::AllowedByLicense,
            public_fixture: PermissionDecision::AllowedByLicense,
        }
    }

    fn permission_evidence() -> Vec<PermissionEvidenceV1> {
        [
            ("license:assessment-clause", SourceRightV1::Assessment),
            ("license:cache-clause", SourceRightV1::Cache),
            ("license:transformation-clause", SourceRightV1::Transformation),
            ("license:redistribution-clause", SourceRightV1::Redistribution),
            ("license:commercial-use-clause", SourceRightV1::CommercialUse),
            ("license:public-fixture-clause", SourceRightV1::PublicFixture),
        ]
        .into_iter()
        .map(|(evidence_id, right)| PermissionEvidenceV1 {
            evidence_id: evidence_id.to_owned(),
            source_id: "fixture:food-table-v1".to_owned(),
            right,
            basis: PermissionEvidenceBasisV1::LicenseTerms,
            evidence_uri: "https://example.invalid/license".to_owned(),
            reviewed_on: "2026-01-01".to_owned(),
            valid_until: "2027-01-01".to_owned(),
            lifecycle: PermissionEvidenceLifecycleV1::Active,
        })
        .collect()
    }

    fn manifest() -> NutritionSourceManifestV1 {
        NutritionSourceManifestV1 {
            schema_version: NUTRITION_SOURCE_SCHEMA_VERSION,
            source_id: "fixture:food-table-v1".to_owned(),
            publisher: "Synthetic test publisher".to_owned(),
            dataset_title: "Synthetic food composition fixture".to_owned(),
            dataset_edition: "v1".to_owned(),
            source_uri: "https://example.invalid/food-table".to_owned(),
            terms_uri: "https://example.invalid/terms".to_owned(),
            license_id: "synthetic-fixture-only".to_owned(),
            permission_references: SourcePermissionReferencesV1 {
                use_for_assessment: Some("license:assessment-clause".to_owned()),
                cache: Some("license:cache-clause".to_owned()),
                transform: Some("license:transformation-clause".to_owned()),
                redistribute: Some("license:redistribution-clause".to_owned()),
                commercial_use: Some("license:commercial-use-clause".to_owned()),
                public_fixture: Some("license:public-fixture-clause".to_owned()),
            },
            permission_evidence: permission_evidence(),
            permissions: allowed_permissions(),
            artifact_sha256: Some(sha256_hex(ARTIFACT)),
            retrieved_on: "2026-01-02".to_owned(),
            reviewed_on: "2026-01-01".to_owned(),
            review_valid_until: "2027-01-01".to_owned(),
            lifecycle: SourceLifecycle::Active,
        }
    }

    // The fixture values and publisher are synthetic. The asserted evidence class
    // exercises the declared-source branch only; this test does not authenticate data.
    fn fact() -> NutritionFoodFactV1 {
        NutritionFoodFactV1 {
            schema_version: NUTRITION_SOURCE_SCHEMA_VERSION,
            source_id: "fixture:food-table-v1".to_owned(),
            source_record_id: "row-001".to_owned(),
            source_record_citation: "Synthetic fixture row 001".to_owned(),
            food_id: "food:test-leaf".to_owned(),
            food_description: "Synthetic leafy food".to_owned(),
            food_taxon_ref: Some("taxon:synthetic-leaf".to_owned()),
            cultivar_id: None,
            identity_confidence: FoodIdentityConfidence::Reviewed,
            food_state_id: "cooked_edible".to_owned(),
            match_quality: FoodMatchQuality::ExactReviewedMatch,
            geographic_match: FoodGeographicMatch::SouthAfricanLocal,
            evidence_class: EvidenceClass::SyntheticFixture,
            nutrient_id: "iron".to_owned(),
            nutrient_definition_ref: "nutrient-definition:iron-v1".to_owned(),
            unit: NutrientUnit::Milligrams,
            value_scaled: Some(1250),
            value_scale: 1000,
            basis_grams: 100,
            quality_notes: Vec::new(),
            transformation_applied: false,
            transformation_lineage: Vec::new(),
        }
    }

    fn use_request() -> SourceUseRequestV1 {
        SourceUseRequestV1 {
            retain_source_data: true,
            transform_record: false,
            commercial_use: false,
            public_fixture: false,
        }
    }

    fn assess(
        manifest: &NutritionSourceManifestV1,
        fact: &NutritionFoodFactV1,
        bytes: Option<&[u8]>,
    ) -> NutritionRegistryAssessmentV1 {
        assess_nutrition_source(manifest, fact, EVALUATION_DATE, &use_request(), bytes)
    }

    #[test]
    fn licensed_current_exact_match_requires_verified_artifact_bytes() {
        let result = assess(&manifest(), &fact(), Some(ARTIFACT));
        assert_eq!(result.source_eligibility, Eligibility::Eligible);
        assert_eq!(result.license_eligibility, Eligibility::Eligible);
        assert_eq!(result.data_quality_eligibility, DataQualityEligibility::ExactMatch);
        assert_eq!(result.artifact_integrity, ArtifactIntegrity::Verified);
        assert_eq!(
            result.nutrition_assessability,
            NutritionAssessability::SyntheticFixtureOnly
        );
        assert!(result.issues.is_empty());
    }

    #[test]
    fn hash_like_text_without_the_artifact_is_not_verification() {
        let result = assess(&manifest(), &fact(), None);
        assert_eq!(result.artifact_integrity, ArtifactIntegrity::Unverified);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::ArtifactDigestUnverified
        }));
    }

    #[test]
    fn wrong_artifact_bytes_block_even_when_the_declared_digest_looks_valid() {
        let result = assess(&manifest(), &fact(), Some(b"different artifact"));
        assert_eq!(result.artifact_integrity, ArtifactIntegrity::Invalid);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::ArtifactDigestMismatch
        }));
    }

    #[test]
    fn retained_artifact_requires_a_digest_and_bytes() {
        let mut m = manifest();
        m.artifact_sha256 = None;
        let result = assess(&m, &fact(), None);
        assert_eq!(result.artifact_integrity, ArtifactIntegrity::Invalid);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::MissingArtifactDigest
        }));
    }

    #[test]
    fn unknown_assessment_permission_fails_closed() {
        let mut m = manifest();
        m.permissions.use_for_assessment = PermissionDecision::Unknown;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::AssessmentUseNotPermitted
        }));
    }

    #[test]
    fn unrelated_permission_requirement_does_not_block_current_use() {
        let mut m = manifest();
        m.permissions.transform = PermissionDecision::RequiresWrittenPermission;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Eligible);
        assert_eq!(
            result.nutrition_assessability,
            NutritionAssessability::SyntheticFixtureOnly
        );
        assert!(result.issues.is_empty());
    }

    #[test]
    fn unresolved_nonempty_permission_reference_fails_closed() {
        let mut m = manifest();
        m.permission_references.use_for_assessment = Some("license:not-in-manifest".to_owned());
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::PermissionEvidenceNotFound
        }));
    }

    #[test]
    fn permission_evidence_for_another_right_cannot_be_reused() {
        let mut m = manifest();
        m.permission_references.commercial_use =
            Some("license:assessment-clause".to_owned());
        let mut use_case = use_request();
        use_case.commercial_use = true;
        let result = assess_nutrition_source(
            &m, &fact(), EVALUATION_DATE, &use_case, Some(ARTIFACT),
        );
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::PermissionEvidenceRightMismatch
        }));
    }

    #[test]
    fn expired_permission_evidence_blocks_the_assessment() {
        let mut m = manifest();
        m.permission_evidence[0].valid_until = "2026-10-14".to_owned();
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::PermissionEvidenceExpired
        }));
    }

    #[test]
    fn revoked_permission_evidence_blocks_the_assessment() {
        let mut m = manifest();
        m.permission_evidence[0].lifecycle = PermissionEvidenceLifecycleV1::Revoked;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::PermissionEvidenceRevoked
        }));
    }

    #[test]
    fn written_permission_must_resolve_to_written_permission_evidence() {
        let mut m = manifest();
        m.permissions.use_for_assessment = PermissionDecision::GrantedByWrittenPermission;
        m.permission_references.use_for_assessment =
            Some("license:assessment-clause".to_owned());
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::PermissionEvidenceBasisMismatch
        }));
    }

    #[test]
    fn a_license_grant_requires_a_right_scoped_license_evidence_reference() {
        let mut m = manifest();
        m.permission_references.cache = None;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::CachePermissionReferenceMissing
        }));
    }

    #[test]
    fn written_permission_requires_a_reference() {
        let mut m = manifest();
        m.permissions.use_for_assessment = PermissionDecision::GrantedByWrittenPermission;
        m.permission_references.use_for_assessment = None;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::AssessmentPermissionReferenceMissing
        }));
    }

    #[test]
    fn a_reference_for_one_right_cannot_authorize_another_right() {
        let mut m = manifest();
        m.permissions.use_for_assessment = PermissionDecision::GrantedByWrittenPermission;
        m.permissions.commercial_use = PermissionDecision::GrantedByWrittenPermission;
        // This reference is specifically for assessment; commercial authorization is absent.
        m.permission_references.use_for_assessment =
            Some("permission:assessment-only".to_owned());
        m.permission_references.commercial_use = None;
        let mut use_case = use_request();
        use_case.commercial_use = true;
        let result = assess_nutrition_source(
            &m,
            &fact(),
            EVALUATION_DATE,
            &use_case,
            Some(ARTIFACT),
        );
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::CommercialUsePermissionReferenceMissing
        }));
        assert!(!result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::AssessmentPermissionReferenceMissing
        }));
    }

    #[test]
    fn cache_written_permission_requires_a_cache_scoped_reference() {
        let mut m = manifest();
        m.permissions.cache = PermissionDecision::GrantedByWrittenPermission;
        m.permission_references.cache = None;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::CachePermissionReferenceMissing
        }));
    }

    #[test]
    fn transformation_written_permission_requires_a_transformation_scoped_reference() {
        let mut m = manifest();
        m.permissions.transform = PermissionDecision::GrantedByWrittenPermission;
        m.permission_references.transform = None;
        let mut f = fact();
        f.transformation_applied = true;
        f.transformation_lineage = vec![NutritionTransformationStepV1 {
            transformation_id: "fixture:cook-001".to_owned(),
            from_food_state_id: "raw_edible".to_owned(),
            to_food_state_id: "cooked_edible".to_owned(),
            method_ref: "fixture:boiling-method-001".to_owned(),
            evidence_ref: "fixture:cooking-evidence-001".to_owned(),
        }];
        let result = assess(&m, &f, Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::TransformationPermissionReferenceMissing
        }));
    }

    #[test]
    fn public_fixture_written_permission_requires_a_fixture_scoped_reference() {
        let mut m = manifest();
        m.permissions.public_fixture = PermissionDecision::GrantedByWrittenPermission;
        m.permission_references.public_fixture = None;
        let mut use_case = use_request();
        use_case.public_fixture = true;
        let result = assess_nutrition_source(
            &m,
            &fact(),
            EVALUATION_DATE,
            &use_case,
            Some(ARTIFACT),
        );
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::PublicFixturePermissionReferenceMissing
        }));
    }

    #[test]
    fn redistribution_written_permission_requires_a_redistribution_scoped_reference() {
        let mut m = manifest();
        m.permissions.redistribute = PermissionDecision::GrantedByWrittenPermission;
        m.permission_references.redistribute = None;
        let mut use_case = use_request();
        use_case.public_fixture = true;
        let result = assess_nutrition_source(
            &m,
            &fact(),
            EVALUATION_DATE,
            &use_case,
            Some(ARTIFACT),
        );
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::RedistributionPermissionReferenceMissing
        }));
    }

    #[test]
    fn public_fixture_requires_both_public_fixture_and_redistribution_rights() {
        let mut m = manifest();
        m.permissions.public_fixture = PermissionDecision::AllowedByLicense;
        m.permissions.redistribute = PermissionDecision::Denied;
        let mut use_case = use_request();
        use_case.public_fixture = true;
        let result = assess_nutrition_source(
            &m,
            &fact(),
            EVALUATION_DATE,
            &use_case,
            Some(ARTIFACT),
        );
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::RedistributionNotPermitted
        }));
    }

    #[test]
    fn no_digest_claim_is_explicit_when_data_are_not_retained() {
        let mut m = manifest();
        m.artifact_sha256 = None;
        let mut use_case = use_request();
        use_case.retain_source_data = false;
        let result = assess_nutrition_source(
            &m,
            &fact(),
            EVALUATION_DATE,
            &use_case,
            None,
        );
        assert_eq!(result.artifact_integrity, ArtifactIntegrity::NoDigestClaimed);
        assert_eq!(
            result.nutrition_assessability,
            NutritionAssessability::SyntheticFixtureOnly
        );
        assert!(result.issues.is_empty());
    }

    #[test]
    fn retention_requires_cache_permission() {
        let mut m = manifest();
        m.permissions.cache = PermissionDecision::Denied;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::CacheNotPermitted
        }));
    }

    #[test]
    fn public_fixture_implies_retention_and_requires_verified_artifact() {
        let mut m = manifest();
        m.artifact_sha256 = None;
        let mut use_case = use_request();
        use_case.retain_source_data = false;
        use_case.public_fixture = true;
        let result = assess_nutrition_source(
            &m,
            &fact(),
            EVALUATION_DATE,
            &use_case,
            None,
        );
        assert_eq!(result.artifact_integrity, ArtifactIntegrity::Invalid);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::MissingArtifactDigest
        }));
    }

    #[test]
    fn transform_requires_explicit_permission() {
        let mut m = manifest();
        let mut f = fact();
        f.transformation_applied = true;
        f.transformation_lineage = vec![NutritionTransformationStepV1 {
            transformation_id: "fixture:cook-001".to_owned(),
            from_food_state_id: "raw_edible".to_owned(),
            to_food_state_id: "cooked_edible".to_owned(),
            method_ref: "fixture:boiling-method-001".to_owned(),
            evidence_ref: "fixture:cooking-evidence-001".to_owned(),
        }];
        m.permissions.transform = PermissionDecision::RequiresWrittenPermission;
        let result = assess(&m, &f, Some(ARTIFACT));
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::TransformationNotPermitted
        }));
    }

    #[test]
    fn future_retrieval_date_blocks_source_record() {
        let mut m = manifest();
        m.retrieved_on = "2026-10-16".to_owned();
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.source_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::SourceRetrievedInFuture
        }));
    }

    #[test]
    fn expired_source_review_blocks_numeric_use() {
        let mut m = manifest();
        m.review_valid_until = "2026-10-14".to_owned();
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.source_eligibility, Eligibility::Blocked);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::ReviewExpired
        }));
    }

    #[test]
    fn revoked_source_cannot_be_used_even_with_a_valid_digest() {
        let mut m = manifest();
        m.lifecycle = SourceLifecycle::Revoked;
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.source_eligibility, Eligibility::Blocked);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::SourceRevoked
        }));
    }

    #[test]
    fn missing_nutrient_value_is_not_assessable_not_zero() {
        let mut f = fact();
        f.value_scaled = None;
        let result = assess(&manifest(), &f, Some(ARTIFACT));
        assert_eq!(result.data_quality_eligibility, DataQualityEligibility::ExactMatch);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::NotAssessable);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn mismatched_source_and_unresolved_food_identity_block() {
        let mut f = fact();
        f.source_id = "another-source".to_owned();
        f.match_quality = FoodMatchQuality::Unresolved;
        let result = assess(&manifest(), &f, Some(ARTIFACT));
        assert_eq!(result.data_quality_eligibility, DataQualityEligibility::Blocked);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::FactSourceMismatch
        }));
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::UnresolvedFoodMatch
        }));
    }

    #[test]
    fn transformation_chain_must_connect_each_preparation_state() {
        let mut f = fact();
        f.transformation_applied = true;
        f.transformation_lineage = vec![
            NutritionTransformationStepV1 {
                transformation_id: "fixture:step-001".to_owned(),
                from_food_state_id: "raw_edible".to_owned(),
                to_food_state_id: "boiled_edible".to_owned(),
                method_ref: "fixture:method-001".to_owned(),
                evidence_ref: "fixture:evidence-001".to_owned(),
            },
            NutritionTransformationStepV1 {
                transformation_id: "fixture:step-002".to_owned(),
                from_food_state_id: "raw_edible".to_owned(),
                to_food_state_id: "cooked_edible".to_owned(),
                method_ref: "fixture:method-002".to_owned(),
                evidence_ref: "fixture:evidence-002".to_owned(),
            },
        ];
        let result = assess(&manifest(), &f, Some(ARTIFACT));
        assert_eq!(result.data_quality_eligibility, DataQualityEligibility::Blocked);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::TransformationChainMismatch
        }));
    }

    #[test]
    fn transformed_food_requires_an_inspectable_state_lineage() {
        let mut f = fact();
        f.transformation_applied = true;
        let result = assess(&manifest(), &f, Some(ARTIFACT));
        assert_eq!(result.data_quality_eligibility, DataQualityEligibility::Blocked);
        assert_eq!(result.nutrition_assessability, NutritionAssessability::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::MissingTransformationLineage
        }));
    }

    #[test]
    fn approximate_geographic_match_is_a_visible_fallback() {
        let mut f = fact();
        f.match_quality = FoodMatchQuality::RegionalMatch;
        f.geographic_match = FoodGeographicMatch::GlobalFallback;
        f.identity_confidence = FoodIdentityConfidence::Probable;
        let result = assess(&manifest(), &f, Some(ARTIFACT));
        assert_eq!(
            result.data_quality_eligibility,
            DataQualityEligibility::QualifiedFallback
        );
        assert_eq!(
            result.nutrition_assessability,
            NutritionAssessability::SyntheticFixtureOnly
        );
    }

    #[test]
    fn malformed_retrieval_date_blocks_source_record() {
        let mut m = manifest();
        m.retrieved_on = "2026-2-1".to_owned();
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.source_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::InvalidRetrievedOn
        }));
    }

    #[test]
    fn synthetic_fixtures_are_never_mislabeled_as_empirical_evidence() {
        let mut f = fact();
        f.evidence_class = EvidenceClass::SyntheticFixture;
        let result = assess(&manifest(), &f, Some(ARTIFACT));
        assert_eq!(
            result.nutrition_assessability,
            NutritionAssessability::SyntheticFixtureOnly
        );
    }

    #[test]
    fn commercial_intent_requires_commercial_permission() {
        let mut m = manifest();
        m.permissions.commercial_use = PermissionDecision::Unknown;
        let mut use_case = use_request();
        use_case.commercial_use = true;
        let result = assess_nutrition_source(
            &m,
            &fact(),
            EVALUATION_DATE,
            &use_case,
            Some(ARTIFACT),
        );
        assert_eq!(result.license_eligibility, Eligibility::Blocked);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::CommercialUseNotPermitted
        }));
    }

    #[test]
    fn malformed_dates_and_noncanonical_digest_block() {
        let mut m = manifest();
        m.reviewed_on = "2026-1-1".to_owned();
        m.artifact_sha256 = Some("A".repeat(64));
        let result = assess(&m, &fact(), Some(ARTIFACT));
        assert_eq!(result.source_eligibility, Eligibility::Blocked);
        assert_eq!(result.artifact_integrity, ArtifactIntegrity::Invalid);
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::InvalidReviewedOn
        }));
        assert!(result.issues.iter().any(|issue| {
            issue.code == RegistryIssueCode::InvalidArtifactDigest
        }));
    }

    #[test]
    fn registry_contract_has_a_deterministic_json_roundtrip() {
        let m = manifest();
        let encoded = serde_json::to_string(&m).expect("manifest serializes");
        let decoded: NutritionSourceManifestV1 =
            serde_json::from_str(&encoded).expect("manifest deserializes");
        assert_eq!(decoded, m);

        let result = assess(&m, &fact(), Some(ARTIFACT));
        let encoded_result = serde_json::to_string(&result).expect("assessment serializes");
        let decoded_result: NutritionRegistryAssessmentV1 =
            serde_json::from_str(&encoded_result).expect("assessment deserializes");
        assert_eq!(decoded_result, result);
    }
}
