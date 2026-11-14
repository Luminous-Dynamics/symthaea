//! Fail-closed policy evaluation over human-reviewed music-component rights records.
//!
//! This module does not parse license prose, authenticate issuers, interpret
//! legal ambiguity, or give legal advice. It evaluates explicit permissions
//! supplied by a qualified human review process. Unknown remains blocked.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const MUSIC_RIGHTS_BUNDLE_SCHEMA_V1: &str = "mel-music-component-rights-v1";
pub const MUSIC_RIGHTS_REQUEST_SCHEMA_V1: &str = "mel-music-rights-use-request-v1";
pub const MUSIC_RIGHTS_DECISION_SCHEMA_V1: &str = "mel-music-rights-decision-v1";
pub const MUSIC_RIGHTS_POLICY_VERSION_V1: &str = "mel-music-rights-policy-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsPermissionV1 {
    Allowed,
    Denied,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsReviewStatusV1 {
    HumanReviewed,
    MachineExtracted,
    Unreviewed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShareAlikeStatusV1 {
    NotApplicable,
    Applies,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsPurposeV1 {
    InternalResearch,
    Evaluation,
    CommercialEvaluation,
    Training,
    FineTuning,
    Redistribution,
    PublicRelease,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RightsPermissionsV1 {
    pub internal_research: RightsPermissionV1,
    pub evaluation: RightsPermissionV1,
    pub training: RightsPermissionV1,
    pub fine_tuning: RightsPermissionV1,
    pub commercial_use: RightsPermissionV1,
    pub redistribution: RightsPermissionV1,
    pub public_release: RightsPermissionV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentLicenseV1 {
    pub license_id: String,
    pub license_version: String,
    pub source_uri: String,
    pub license_text_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rights_holder_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieved_at_utc: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RightsHumanReviewV1 {
    /// Opaque reviewer reference; do not put personal contact details here.
    pub reviewer_ref: String,
    pub reviewer_role: String,
    pub jurisdiction: String,
    pub reviewed_at_utc: String,
    pub rationale: String,
    /// Digest of the human review record, not a signature or identity proof.
    pub decision_record_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentRightsV1 {
    pub component_id: String,
    pub component_sha256: String,
    pub license: ComponentLicenseV1,
    pub permissions: RightsPermissionsV1,
    pub attribution_required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution_statement: Option<String>,
    pub notice_required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notice_statement: Option<String>,
    pub share_alike: ShareAlikeStatusV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub share_alike_scope: Option<String>,
    pub review_status: RightsReviewStatusV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_review: Option<RightsHumanReviewV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicComponentRightsBundleV1 {
    pub schema_version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    /// Components must be sorted by component_id and have unique IDs.
    pub components: Vec<ComponentRightsV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRightsUseRequestV1 {
    pub schema_version: String,
    pub purpose: RightsPurposeV1,
    /// Explicit declaration from the caller; this policy does not infer the
    /// legal character of an organization's activity.
    pub commercial_context: bool,
    pub distribute_output: bool,
    pub public_release: bool,
    pub derivative_created: bool,
    pub output_attribution_attached: bool,
    pub output_notices_attached: bool,
    pub output_share_alike_declared: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsDecisionStatusV1 {
    Allowed,
    Denied,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsDecisionReasonCodeV1 {
    UnreviewedComponent,
    MissingHumanReviewReceipt,
    MachineExtractionIsNotLegalReview,
    PermissionUnknown,
    PermissionDenied,
    AttributionNotAttached,
    NoticeNotAttached,
    ShareAlikeStatusUnknown,
    ShareAlikeNotDeclared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RightsDecisionReasonV1 {
    pub component_id: String,
    pub code: RightsDecisionReasonCodeV1,
    pub purpose: Option<RightsPurposeV1>,
    pub permission: Option<RightsPermissionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RightsOutputObligationV1 {
    pub component_id: String,
    pub attribution_statement: Option<String>,
    pub notice_statement: Option<String>,
    pub share_alike_scope: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicRightsDecisionReceiptV1 {
    pub schema_version: String,
    pub policy_version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub rights_bundle_sha256: String,
    pub use_request_sha256: String,
    pub status: RightsDecisionStatusV1,
    pub reasons: Vec<RightsDecisionReasonV1>,
    pub output_obligations: Vec<RightsOutputObligationV1>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RightsPolicyIssueV1 {
    WrongBundleSchema,
    WrongRequestSchema,
    EmptyDatasetId,
    EmptyDatasetVersion,
    EmptyComponents,
    EmptyComponentId,
    DuplicateComponentId,
    NonCanonicalComponentOrder,
    InvalidComponentDigest { component_id: String },
    EmptyLicenseField { component_id: String, field: &'static str },
    InvalidLicenseDigest { component_id: String },
    InvalidHumanReviewRecord { component_id: String },
    ReviewRecordStatusMismatch { component_id: String },
    MissingAttributionStatement { component_id: String },
    MissingNoticeStatement { component_id: String },
    MissingShareAlikeScope { component_id: String },
    CommercialEvaluationWithoutCommercialContext,
    PublicReleaseRequiresDistribution,
    SerializationError,
    ReceiptMismatch,
}

#[derive(Serialize)]
struct RightsCommitmentV1<'a, T> {
    domain: &'a str,
    value: &'a T,
}

fn rights_digest_v1<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, RightsPolicyIssueV1> {
    let bytes = serde_json::to_vec(&RightsCommitmentV1 { domain, value })
        .map_err(|_| RightsPolicyIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous.music-rights-policy.v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn valid_rights_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn validate_music_component_rights_bundle_v1(
    bundle: &MusicComponentRightsBundleV1,
) -> Vec<RightsPolicyIssueV1> {
    let mut issues = Vec::new();
    if bundle.schema_version != MUSIC_RIGHTS_BUNDLE_SCHEMA_V1 {
        issues.push(RightsPolicyIssueV1::WrongBundleSchema);
    }
    if bundle.dataset_id.trim().is_empty() {
        issues.push(RightsPolicyIssueV1::EmptyDatasetId);
    }
    if bundle.dataset_version.trim().is_empty() {
        issues.push(RightsPolicyIssueV1::EmptyDatasetVersion);
    }
    if bundle.components.is_empty() {
        issues.push(RightsPolicyIssueV1::EmptyComponents);
        return issues;
    }

    let mut seen = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for component in &bundle.components {
        if component.component_id.trim().is_empty() {
            issues.push(RightsPolicyIssueV1::EmptyComponentId);
        }
        if !seen.insert(component.component_id.as_str()) {
            issues.push(RightsPolicyIssueV1::DuplicateComponentId);
        }
        if previous.is_some_and(|value| value > component.component_id.as_str()) {
            issues.push(RightsPolicyIssueV1::NonCanonicalComponentOrder);
        }
        previous = Some(component.component_id.as_str());

        if !valid_rights_sha256(&component.component_sha256) {
            issues.push(RightsPolicyIssueV1::InvalidComponentDigest {
                component_id: component.component_id.clone(),
            });
        }
        for (field, value) in [
            ("license_id", component.license.license_id.as_str()),
            ("license_version", component.license.license_version.as_str()),
            ("source_uri", component.license.source_uri.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(RightsPolicyIssueV1::EmptyLicenseField {
                    component_id: component.component_id.clone(),
                    field,
                });
            }
        }
        if !valid_rights_sha256(&component.license.license_text_sha256) {
            issues.push(RightsPolicyIssueV1::InvalidLicenseDigest {
                component_id: component.component_id.clone(),
            });
        }
        if let Some(review) = component.human_review.as_ref() {
            let review_fields = [
                review.reviewer_ref.as_str(),
                review.reviewer_role.as_str(),
                review.jurisdiction.as_str(),
                review.reviewed_at_utc.as_str(),
                review.rationale.as_str(),
            ];
            if review_fields.iter().any(|value| value.trim().is_empty())
                || !valid_rights_sha256(&review.decision_record_sha256)
            {
                issues.push(RightsPolicyIssueV1::InvalidHumanReviewRecord {
                    component_id: component.component_id.clone(),
                });
            }
            if component.review_status != RightsReviewStatusV1::HumanReviewed {
                issues.push(RightsPolicyIssueV1::ReviewRecordStatusMismatch {
                    component_id: component.component_id.clone(),
                });
            }
        }
        if component.attribution_required
            && component.attribution_statement.as_deref().is_none_or(str::is_empty)
        {
            issues.push(RightsPolicyIssueV1::MissingAttributionStatement {
                component_id: component.component_id.clone(),
            });
        }
        if component.notice_required
            && component.notice_statement.as_deref().is_none_or(str::is_empty)
        {
            issues.push(RightsPolicyIssueV1::MissingNoticeStatement {
                component_id: component.component_id.clone(),
            });
        }
        if component.share_alike == ShareAlikeStatusV1::Applies
            && component.share_alike_scope.as_deref().is_none_or(str::is_empty)
        {
            issues.push(RightsPolicyIssueV1::MissingShareAlikeScope {
                component_id: component.component_id.clone(),
            });
        }
    }
    issues
}

fn permission_for_v1(
    permissions: &RightsPermissionsV1,
    purpose: RightsPurposeV1,
) -> RightsPermissionV1 {
    match purpose {
        RightsPurposeV1::InternalResearch => permissions.internal_research,
        RightsPurposeV1::Evaluation => permissions.evaluation,
        RightsPurposeV1::CommercialEvaluation => permissions.commercial_use,
        RightsPurposeV1::Training => permissions.training,
        RightsPurposeV1::FineTuning => permissions.fine_tuning,
        RightsPurposeV1::Redistribution => permissions.redistribution,
        RightsPurposeV1::PublicRelease => permissions.public_release,
    }
}

fn required_purposes_v1(request: &MusicRightsUseRequestV1) -> Vec<RightsPurposeV1> {
    let mut required = Vec::new();
    let push_unique = |items: &mut Vec<RightsPurposeV1>, item| {
        if !items.contains(&item) {
            items.push(item);
        }
    };
    match request.purpose {
        RightsPurposeV1::InternalResearch => {
            push_unique(&mut required, RightsPurposeV1::InternalResearch);
        }
        RightsPurposeV1::Evaluation => {
            push_unique(&mut required, RightsPurposeV1::Evaluation);
        }
        RightsPurposeV1::CommercialEvaluation => {
            push_unique(&mut required, RightsPurposeV1::Evaluation);
            push_unique(&mut required, RightsPurposeV1::CommercialEvaluation);
        }
        RightsPurposeV1::Training => push_unique(&mut required, RightsPurposeV1::Training),
        RightsPurposeV1::FineTuning => push_unique(&mut required, RightsPurposeV1::FineTuning),
        RightsPurposeV1::Redistribution => {
            push_unique(&mut required, RightsPurposeV1::Redistribution);
        }
        RightsPurposeV1::PublicRelease => {
            push_unique(&mut required, RightsPurposeV1::Redistribution);
            push_unique(&mut required, RightsPurposeV1::PublicRelease);
        }
    }
    if request.commercial_context {
        push_unique(&mut required, RightsPurposeV1::CommercialEvaluation);
    }
    if request.distribute_output {
        push_unique(&mut required, RightsPurposeV1::Redistribution);
    }
    if request.public_release {
        push_unique(&mut required, RightsPurposeV1::PublicRelease);
    }
    required
}

fn receipt_digest_v1(
    receipt: &MusicRightsDecisionReceiptV1,
) -> Result<String, RightsPolicyIssueV1> {
    #[derive(Serialize)]
    struct Commitment<'a> {
        schema_version: &'a str,
        policy_version: &'a str,
        dataset_id: &'a str,
        dataset_version: &'a str,
        rights_bundle_sha256: &'a str,
        use_request_sha256: &'a str,
        status: RightsDecisionStatusV1,
        reasons: &'a [RightsDecisionReasonV1],
        output_obligations: &'a [RightsOutputObligationV1],
    }
    let commitment = Commitment {
        schema_version: &receipt.schema_version,
        policy_version: &receipt.policy_version,
        dataset_id: &receipt.dataset_id,
        dataset_version: &receipt.dataset_version,
        rights_bundle_sha256: &receipt.rights_bundle_sha256,
        use_request_sha256: &receipt.use_request_sha256,
        status: receipt.status,
        reasons: &receipt.reasons,
        output_obligations: &receipt.output_obligations,
    };
    rights_digest_v1("decision-receipt", &commitment)
}

/// Evaluates explicit component permissions. Every component must be human
/// reviewed and explicitly permit every applicable purpose. Missing, unknown,
/// or machine-extracted-only terms block admission; any explicit denial denies.
pub fn evaluate_music_rights_use_v1(
    bundle: &MusicComponentRightsBundleV1,
    request: &MusicRightsUseRequestV1,
) -> Result<MusicRightsDecisionReceiptV1, Vec<RightsPolicyIssueV1>> {
    let mut structural = validate_music_component_rights_bundle_v1(bundle);
    if request.schema_version != MUSIC_RIGHTS_REQUEST_SCHEMA_V1 {
        structural.push(RightsPolicyIssueV1::WrongRequestSchema);
    }
    if request.purpose == RightsPurposeV1::CommercialEvaluation && !request.commercial_context {
        structural.push(RightsPolicyIssueV1::CommercialEvaluationWithoutCommercialContext);
    }
    if request.public_release && !request.distribute_output {
        structural.push(RightsPolicyIssueV1::PublicReleaseRequiresDistribution);
    }
    if !structural.is_empty() {
        return Err(structural);
    }

    let mut reasons = Vec::new();
    let mut obligations = Vec::new();
    let required = required_purposes_v1(request);

    for component in &bundle.components {
        match component.review_status {
            RightsReviewStatusV1::HumanReviewed => {
                if component.human_review.is_none() {
                    reasons.push(RightsDecisionReasonV1 {
                        component_id: component.component_id.clone(),
                        code: RightsDecisionReasonCodeV1::MissingHumanReviewReceipt,
                        purpose: None,
                        permission: None,
                    });
                }
            }
            RightsReviewStatusV1::MachineExtracted => {
                reasons.push(RightsDecisionReasonV1 {
                    component_id: component.component_id.clone(),
                    code: RightsDecisionReasonCodeV1::MachineExtractionIsNotLegalReview,
                    purpose: None,
                    permission: None,
                });
            }
            RightsReviewStatusV1::Unreviewed => {
                reasons.push(RightsDecisionReasonV1 {
                    component_id: component.component_id.clone(),
                    code: RightsDecisionReasonCodeV1::UnreviewedComponent,
                    purpose: None,
                    permission: None,
                });
            }
        }

        for purpose in &required {
            let permission = permission_for_v1(&component.permissions, *purpose);
            let code = match permission {
                RightsPermissionV1::Allowed => None,
                RightsPermissionV1::Denied => Some(RightsDecisionReasonCodeV1::PermissionDenied),
                RightsPermissionV1::Unknown => Some(RightsDecisionReasonCodeV1::PermissionUnknown),
            };
            if let Some(code) = code {
                reasons.push(RightsDecisionReasonV1 {
                    component_id: component.component_id.clone(),
                    code,
                    purpose: Some(*purpose),
                    permission: Some(permission),
                });
            }
        }

        if request.distribute_output
            && component.attribution_required
            && !request.output_attribution_attached
        {
            reasons.push(RightsDecisionReasonV1 {
                component_id: component.component_id.clone(),
                code: RightsDecisionReasonCodeV1::AttributionNotAttached,
                purpose: None,
                permission: None,
            });
        }
        if request.distribute_output
            && component.notice_required
            && !request.output_notices_attached
        {
            reasons.push(RightsDecisionReasonV1 {
                component_id: component.component_id.clone(),
                code: RightsDecisionReasonCodeV1::NoticeNotAttached,
                purpose: None,
                permission: None,
            });
        }
        if request.distribute_output && request.derivative_created {
            match component.share_alike {
                ShareAlikeStatusV1::NotApplicable => {}
                ShareAlikeStatusV1::Unknown => reasons.push(RightsDecisionReasonV1 {
                    component_id: component.component_id.clone(),
                    code: RightsDecisionReasonCodeV1::ShareAlikeStatusUnknown,
                    purpose: None,
                    permission: None,
                }),
                ShareAlikeStatusV1::Applies if !request.output_share_alike_declared => {
                    reasons.push(RightsDecisionReasonV1 {
                        component_id: component.component_id.clone(),
                        code: RightsDecisionReasonCodeV1::ShareAlikeNotDeclared,
                        purpose: None,
                        permission: None,
                    });
                }
                ShareAlikeStatusV1::Applies => {}
            }
        }

        obligations.push(RightsOutputObligationV1 {
            component_id: component.component_id.clone(),
            attribution_statement: if component.attribution_required {
                component.attribution_statement.clone()
            } else {
                None
            },
            notice_statement: if component.notice_required {
                component.notice_statement.clone()
            } else {
                None
            },
            share_alike_scope: if component.share_alike == ShareAlikeStatusV1::Applies {
                component.share_alike_scope.clone()
            } else {
                None
            },
        });
    }

    let status = if reasons.iter().any(|reason| {
        reason.code == RightsDecisionReasonCodeV1::PermissionDenied
    }) {
        RightsDecisionStatusV1::Denied
    } else if !reasons.is_empty() {
        RightsDecisionStatusV1::Blocked
    } else {
        RightsDecisionStatusV1::Allowed
    };

    let rights_bundle_sha256 =
        rights_digest_v1("component-rights-bundle", bundle).map_err(|issue| vec![issue])?;
    let use_request_sha256 =
        rights_digest_v1("use-request", request).map_err(|issue| vec![issue])?;
    let mut receipt = MusicRightsDecisionReceiptV1 {
        schema_version: MUSIC_RIGHTS_DECISION_SCHEMA_V1.to_owned(),
        policy_version: MUSIC_RIGHTS_POLICY_VERSION_V1.to_owned(),
        dataset_id: bundle.dataset_id.clone(),
        dataset_version: bundle.dataset_version.clone(),
        rights_bundle_sha256,
        use_request_sha256,
        status,
        reasons,
        output_obligations: obligations,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest_v1(&receipt).map_err(|issue| vec![issue])?;
    Ok(receipt)
}

/// Recompute the policy decision and compare the whole receipt. This verifies
/// deterministic policy replay, not the authority or truth of the human review.
pub fn verify_music_rights_decision_receipt_v1(
    receipt: &MusicRightsDecisionReceiptV1,
    bundle: &MusicComponentRightsBundleV1,
    request: &MusicRightsUseRequestV1,
) -> Vec<RightsPolicyIssueV1> {
    match evaluate_music_rights_use_v1(bundle, request) {
        Ok(expected) if &expected == receipt => Vec::new(),
        _ => vec![RightsPolicyIssueV1::ReceiptMismatch],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPONENT_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const LICENSE_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const REVIEW_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn permissive_permissions() -> RightsPermissionsV1 {
        RightsPermissionsV1 {
            internal_research: RightsPermissionV1::Allowed,
            evaluation: RightsPermissionV1::Allowed,
            training: RightsPermissionV1::Allowed,
            fine_tuning: RightsPermissionV1::Allowed,
            commercial_use: RightsPermissionV1::Allowed,
            redistribution: RightsPermissionV1::Allowed,
            public_release: RightsPermissionV1::Allowed,
        }
    }

    fn component(
        component_id: &str,
        license_id: &str,
        permissions: RightsPermissionsV1,
    ) -> ComponentRightsV1 {
        ComponentRightsV1 {
            component_id: component_id.to_owned(),
            component_sha256: COMPONENT_SHA.to_owned(),
            license: ComponentLicenseV1 {
                license_id: license_id.to_owned(),
                license_version: "4.0".to_owned(),
                source_uri: "https://example.invalid/license".to_owned(),
                license_text_sha256: LICENSE_SHA.to_owned(),
                rights_holder_ref: Some("synthetic-rights-holder-ref".to_owned()),
                source_revision: Some("synthetic-license-revision-v1".to_owned()),
                retrieved_at_utc: Some("2026-11-14".to_owned()),
                inherited_from: None,
            },
            permissions,
            attribution_required: true,
            attribution_statement: Some(format!("Attribution for {component_id}")),
            notice_required: true,
            notice_statement: Some(format!("Notice for {component_id}")),
            share_alike: ShareAlikeStatusV1::NotApplicable,
            share_alike_scope: None,
            review_status: RightsReviewStatusV1::HumanReviewed,
            human_review: Some(RightsHumanReviewV1 {
                reviewer_ref: "reviewer-ref-001".to_owned(),
                reviewer_role: "rights-reviewer".to_owned(),
                jurisdiction: "review-jurisdiction-recorded".to_owned(),
                reviewed_at_utc: "2026-11-14T07:00:00Z".to_owned(),
                rationale: "Synthetic fixture review; no external license determination".to_owned(),
                decision_record_sha256: REVIEW_SHA.to_owned(),
            }),
        }
    }

    fn bundle(components: Vec<ComponentRightsV1>) -> MusicComponentRightsBundleV1 {
        MusicComponentRightsBundleV1 {
            schema_version: MUSIC_RIGHTS_BUNDLE_SCHEMA_V1.to_owned(),
            dataset_id: "synthetic-rights-fixture".to_owned(),
            dataset_version: "synthetic-v1".to_owned(),
            components,
        }
    }

    fn request(purpose: RightsPurposeV1) -> MusicRightsUseRequestV1 {
        MusicRightsUseRequestV1 {
            schema_version: MUSIC_RIGHTS_REQUEST_SCHEMA_V1.to_owned(),
            purpose,
            commercial_context: purpose == RightsPurposeV1::CommercialEvaluation,
            distribute_output: false,
            public_release: false,
            derivative_created: false,
            output_attribution_attached: false,
            output_notices_attached: false,
            output_share_alike_declared: false,
        }
    }

    #[test]
    fn reviewed_component_allows_explicit_evaluation_and_receipt_replays() {
        let rights = bundle(vec![component(
            "audio",
            "CC-BY-4.0",
            permissive_permissions(),
        )]);
        let use_request = request(RightsPurposeV1::Evaluation);
        let receipt = evaluate_music_rights_use_v1(&rights, &use_request).unwrap();
        assert_eq!(receipt.status, RightsDecisionStatusV1::Allowed);
        assert!(verify_music_rights_decision_receipt_v1(&receipt, &rights, &use_request).is_empty());
    }

    #[test]
    fn noncommercial_component_denies_commercial_evaluation() {
        let mut permissions = permissive_permissions();
        permissions.commercial_use = RightsPermissionV1::Denied;
        let rights = bundle(vec![component(
            "moisesdb-derived-audio",
            "CC-BY-NC-SA-4.0",
            permissions,
        )]);
        let receipt = evaluate_music_rights_use_v1(
            &rights,
            &request(RightsPurposeV1::CommercialEvaluation),
        )
        .unwrap();
        assert_eq!(receipt.status, RightsDecisionStatusV1::Denied);
        assert!(receipt.reasons.iter().any(|reason| {
            reason.component_id == "moisesdb-derived-audio"
                && reason.code == RightsDecisionReasonCodeV1::PermissionDenied
        }));
    }

    #[test]
    fn unknown_permission_blocks_instead_of_defaulting_to_allowed() {
        let mut permissions = permissive_permissions();
        permissions.evaluation = RightsPermissionV1::Unknown;
        let rights = bundle(vec![component("unresolved-source", "UNKNOWN", permissions)]);
        let receipt =
            evaluate_music_rights_use_v1(&rights, &request(RightsPurposeV1::Evaluation)).unwrap();
        assert_eq!(receipt.status, RightsDecisionStatusV1::Blocked);
        assert!(receipt.reasons.iter().any(|reason| {
            reason.code == RightsDecisionReasonCodeV1::PermissionUnknown
        }));
    }

    #[test]
    fn machine_extracted_terms_are_not_human_review() {
        let mut item = component("machine-extracted", "CC-BY-4.0", permissive_permissions());
        item.review_status = RightsReviewStatusV1::MachineExtracted;
        item.human_review = None;
        let rights = bundle(vec![item]);
        let receipt =
            evaluate_music_rights_use_v1(&rights, &request(RightsPurposeV1::Evaluation)).unwrap();
        assert_eq!(receipt.status, RightsDecisionStatusV1::Blocked);
        assert!(receipt.reasons.iter().any(|reason| {
            reason.code == RightsDecisionReasonCodeV1::MachineExtractionIsNotLegalReview
        }));
    }

    #[test]
    fn restrictive_inherited_component_controls_the_whole_bundle() {
        let general = component("dataset-metadata", "CC-BY-4.0", permissive_permissions());
        let mut restricted_permissions = permissive_permissions();
        restricted_permissions.training = RightsPermissionV1::Denied;
        let stem = component("source-stem", "CC-BY-NC-SA-4.0", restricted_permissions);
        let rights = bundle(vec![general, stem]);
        let receipt =
            evaluate_music_rights_use_v1(&rights, &request(RightsPurposeV1::Training)).unwrap();
        assert_eq!(receipt.status, RightsDecisionStatusV1::Denied);
        assert!(receipt.reasons.iter().any(|reason| {
            reason.component_id == "source-stem"
                && reason.code == RightsDecisionReasonCodeV1::PermissionDenied
        }));
    }

    #[test]
    fn distribution_requires_attribution_notice_and_share_alike_declaration() {
        let mut item = component("derived-target", "CC-BY-NC-SA-4.0", permissive_permissions());
        item.share_alike = ShareAlikeStatusV1::Applies;
        item.share_alike_scope = Some("Same license family for derivatives".to_owned());
        let rights = bundle(vec![item]);
        let mut use_request = request(RightsPurposeV1::Evaluation);
        use_request.distribute_output = true;
        use_request.derivative_created = true;

        let blocked = evaluate_music_rights_use_v1(&rights, &use_request).unwrap();
        assert_eq!(blocked.status, RightsDecisionStatusV1::Blocked);
        assert!(blocked.reasons.iter().any(|reason| {
            reason.code == RightsDecisionReasonCodeV1::AttributionNotAttached
        }));
        assert!(blocked.reasons.iter().any(|reason| {
            reason.code == RightsDecisionReasonCodeV1::NoticeNotAttached
        }));
        assert!(blocked.reasons.iter().any(|reason| {
            reason.code == RightsDecisionReasonCodeV1::ShareAlikeNotDeclared
        }));

        use_request.output_attribution_attached = true;
        use_request.output_notices_attached = true;
        use_request.output_share_alike_declared = true;
        let allowed = evaluate_music_rights_use_v1(&rights, &use_request).unwrap();
        assert_eq!(allowed.status, RightsDecisionStatusV1::Allowed);
        assert!(allowed.output_obligations[0].attribution_statement.is_some());
        assert!(allowed.output_obligations[0].notice_statement.is_some());
        assert!(allowed.output_obligations[0].share_alike_scope.is_some());
    }

    #[test]
    fn tampered_receipt_rejects() {
        let rights = bundle(vec![component(
            "audio",
            "CC-BY-4.0",
            permissive_permissions(),
        )]);
        let use_request = request(RightsPurposeV1::Evaluation);
        let mut receipt = evaluate_music_rights_use_v1(&rights, &use_request).unwrap();
        receipt.status = RightsDecisionStatusV1::Denied;
        assert_eq!(
            verify_music_rights_decision_receipt_v1(&receipt, &rights, &use_request),
            vec![RightsPolicyIssueV1::ReceiptMismatch]
        );
    }

    #[test]
    fn duplicate_and_noncanonical_components_reject() {
        let first = component("audio-a", "CC-BY-4.0", permissive_permissions());
        let second = component("audio-b", "CC-BY-4.0", permissive_permissions());
        let reversed = bundle(vec![second.clone(), first.clone()]);
        assert!(validate_music_component_rights_bundle_v1(&reversed)
            .contains(&RightsPolicyIssueV1::NonCanonicalComponentOrder));
        let duplicate = bundle(vec![first.clone(), first]);
        assert!(validate_music_component_rights_bundle_v1(&duplicate)
            .contains(&RightsPolicyIssueV1::DuplicateComponentId));
    }

    #[test]
    fn unknown_share_alike_status_blocks_derivative_distribution() {
        let mut item = component("derived-audio", "UNKNOWN", permissive_permissions());
        item.share_alike = ShareAlikeStatusV1::Unknown;
        let rights = bundle(vec![item]);
        let mut use_request = request(RightsPurposeV1::Evaluation);
        use_request.distribute_output = true;
        use_request.derivative_created = true;
        use_request.output_attribution_attached = true;
        use_request.output_notices_attached = true;
        let receipt = evaluate_music_rights_use_v1(&rights, &use_request).unwrap();
        assert_eq!(receipt.status, RightsDecisionStatusV1::Blocked);
        assert!(receipt.reasons.iter().any(|reason| {
            reason.code == RightsDecisionReasonCodeV1::ShareAlikeStatusUnknown
        }));
    }

    #[test]
    fn human_reviewed_component_without_review_record_is_blocked() {
        let mut item = component("missing-review", "CC-BY-4.0", permissive_permissions());
        item.human_review = None;
        let rights = bundle(vec![item]);
        let receipt =
            evaluate_music_rights_use_v1(&rights, &request(RightsPurposeV1::Evaluation)).unwrap();
        assert_eq!(receipt.status, RightsDecisionStatusV1::Blocked);
        assert!(receipt.reasons.iter().any(|reason| {
            reason.code == RightsDecisionReasonCodeV1::MissingHumanReviewReceipt
        }));
    }

    #[test]
    fn malformed_human_review_record_rejects() {
        let mut item = component("invalid-review", "CC-BY-4.0", permissive_permissions());
        item.human_review.as_mut().unwrap().rationale.clear();
        let rights = bundle(vec![item]);
        assert!(validate_music_component_rights_bundle_v1(&rights).contains(
            &RightsPolicyIssueV1::InvalidHumanReviewRecord {
                component_id: "invalid-review".to_owned()
            }
        ));
    }

    #[test]
    fn public_release_requires_explicit_distribution() {
        let rights = bundle(vec![component(
            "audio",
            "CC-BY-4.0",
            permissive_permissions(),
        )]);
        let mut use_request = request(RightsPurposeV1::PublicRelease);
        use_request.public_release = true;
        assert!(matches!(
            evaluate_music_rights_use_v1(&rights, &use_request),
            Err(issues) if issues.contains(&RightsPolicyIssueV1::PublicReleaseRequiresDistribution)
        ));
    }
}
