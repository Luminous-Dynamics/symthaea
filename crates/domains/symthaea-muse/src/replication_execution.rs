// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sealed execution record for one independent replication site.
//!
//! Site conclusions are derived from the frozen primary endpoint and confidence
//! interval. They are not accepted as investigator-entered labels.

use crate::evidence_digest::canonical_json_sha256;
use crate::replication_package::{ReplicationSitePackage, replication_package_commitment};
use crate::replication_protocol::{
    FavorableDirection, FrozenReplicationProtocol, ReplicationKind, replication_protocol_commitment,
};
use crate::replication_site_registry::{
    ReplicationSiteRegistry, ReplicationSiteStatus, replication_site_registry_commitment,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REPLICATION_EXECUTION_VERSION: &str = "symthaea-muse-replication-execution-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiteReplicationConclusion {
    SupportsReplication,
    DoesNotSupportReplication,
    Inconclusive,
    DescriptiveOnly,
    NonEstimable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationDeviation {
    pub category: String,
    pub description: String,
    pub approved_before_collection: bool,
    pub material_to_primary_claim: bool,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SitePrimaryResult {
    pub endpoint_id: String,
    pub estimate: f64,
    pub standard_error: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub p_value: f64,
    pub participant_count: u32,
    pub family_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicationSiteExecutionRecord {
    pub record_version: String,
    pub replication_id: String,
    pub site_id: String,
    pub protocol_sha256: String,
    pub site_registry_sha256: String,
    pub site_package_sha256: String,
    pub package_receipt_sha256: String,
    pub collection_close_sha256: String,
    pub dataset_sha256: String,
    pub analysis_plan_sha256: String,
    pub primary_analysis_sha256: String,
    pub independent_analysis_sha256: String,
    pub analysis_crosscheck_sha256: String,
    pub execution_environment_sha256: String,
    pub source_outcome_access_audit_sha256: String,
    pub deviations: Vec<ReplicationDeviation>,
    pub primary_result: Option<SitePrimaryResult>,
    pub conclusion: SiteReplicationConclusion,
    pub all_frozen_commands_succeeded: bool,
    pub collection_blinded_until_close: bool,
    pub source_outcomes_withheld_until_close: bool,
    pub public_release_uri: String,
    pub completed_at_utc: String,
    pub record_sha256: String,
}

#[derive(Serialize)]
struct ExecutionCommitment<'a> {
    record_version: &'a str,
    replication_id: &'a str,
    site_id: &'a str,
    protocol_sha256: &'a str,
    site_registry_sha256: &'a str,
    site_package_sha256: &'a str,
    package_receipt_sha256: &'a str,
    collection_close_sha256: &'a str,
    dataset_sha256: &'a str,
    analysis_plan_sha256: &'a str,
    primary_analysis_sha256: &'a str,
    independent_analysis_sha256: &'a str,
    analysis_crosscheck_sha256: &'a str,
    execution_environment_sha256: &'a str,
    source_outcome_access_audit_sha256: &'a str,
    deviations: &'a [ReplicationDeviation],
    primary_result: &'a Option<SitePrimaryResult>,
    conclusion: SiteReplicationConclusion,
    all_frozen_commands_succeeded: bool,
    collection_blinded_until_close: bool,
    source_outcomes_withheld_until_close: bool,
    public_release_uri: &'a str,
    completed_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationExecutionIssue {
    WrongVersion {
        found: String,
    },
    InvalidProtocol,
    InvalidSiteRegistry,
    InvalidPackage,
    AuthorityMismatch,
    UnknownOrInactiveSite {
        site_id: String,
    },
    EmptyField {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    DuplicateDeviation {
        category: String,
    },
    ProtocolDeviationNotMarkedMaterial {
        category: String,
    },
    InvalidPrimaryResult {
        field: String,
    },
    PrimaryEndpointMismatch,
    ConclusionMismatch {
        expected: SiteReplicationConclusion,
        found: SiteReplicationConclusion,
    },
    SerializationFailed,
    RecordDigestMismatch,
}

pub fn replication_execution_commitment(
    record: &ReplicationSiteExecutionRecord,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ExecutionCommitment {
        record_version: &record.record_version,
        replication_id: &record.replication_id,
        site_id: &record.site_id,
        protocol_sha256: &record.protocol_sha256,
        site_registry_sha256: &record.site_registry_sha256,
        site_package_sha256: &record.site_package_sha256,
        package_receipt_sha256: &record.package_receipt_sha256,
        collection_close_sha256: &record.collection_close_sha256,
        dataset_sha256: &record.dataset_sha256,
        analysis_plan_sha256: &record.analysis_plan_sha256,
        primary_analysis_sha256: &record.primary_analysis_sha256,
        independent_analysis_sha256: &record.independent_analysis_sha256,
        analysis_crosscheck_sha256: &record.analysis_crosscheck_sha256,
        execution_environment_sha256: &record.execution_environment_sha256,
        source_outcome_access_audit_sha256: &record.source_outcome_access_audit_sha256,
        deviations: &record.deviations,
        primary_result: &record.primary_result,
        conclusion: record.conclusion,
        all_frozen_commands_succeeded: record.all_frozen_commands_succeeded,
        collection_blinded_until_close: record.collection_blinded_until_close,
        source_outcomes_withheld_until_close: record.source_outcomes_withheld_until_close,
        public_release_uri: &record.public_release_uri,
        completed_at_utc: &record.completed_at_utc,
    })
}

pub fn seal_replication_execution(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    package: &ReplicationSitePackage,
    record: &mut ReplicationSiteExecutionRecord,
) -> Result<(), Vec<ReplicationExecutionIssue>> {
    record
        .deviations
        .sort_by(|left, right| left.category.cmp(&right.category));
    record.conclusion = derive_site_conclusion(protocol, record);
    record.record_sha256 = replication_execution_commitment(record)
        .map_err(|_| vec![ReplicationExecutionIssue::SerializationFailed])?;
    let issues = validate_replication_execution(protocol, registry, package, record);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn derive_site_conclusion(
    protocol: &FrozenReplicationProtocol,
    record: &ReplicationSiteExecutionRecord,
) -> SiteReplicationConclusion {
    if record
        .deviations
        .iter()
        .any(|deviation| deviation.material_to_primary_claim)
        || !record.all_frozen_commands_succeeded
        || !record.collection_blinded_until_close
        || !record.source_outcomes_withheld_until_close
    {
        return SiteReplicationConclusion::DescriptiveOnly;
    }
    let Some(result) = &record.primary_result else {
        return SiteReplicationConclusion::NonEstimable;
    };
    if !valid_primary_numbers(result) {
        return SiteReplicationConclusion::NonEstimable;
    }
    if result.participant_count < protocol.participant_target_per_site
        || result.family_count < protocol.family_target_per_site
    {
        return SiteReplicationConclusion::DescriptiveOnly;
    }
    let margin = protocol.primary_endpoint.practical_margin;
    match protocol.primary_endpoint.favorable_direction {
        FavorableDirection::Higher => {
            if result.confidence_lower >= margin {
                SiteReplicationConclusion::SupportsReplication
            } else if result.confidence_upper < margin {
                SiteReplicationConclusion::DoesNotSupportReplication
            } else {
                SiteReplicationConclusion::Inconclusive
            }
        }
        FavorableDirection::Lower => {
            let threshold = -margin;
            if result.confidence_upper <= threshold {
                SiteReplicationConclusion::SupportsReplication
            } else if result.confidence_lower > threshold {
                SiteReplicationConclusion::DoesNotSupportReplication
            } else {
                SiteReplicationConclusion::Inconclusive
            }
        }
    }
}

pub fn validate_replication_execution(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    package: &ReplicationSitePackage,
    record: &ReplicationSiteExecutionRecord,
) -> Vec<ReplicationExecutionIssue> {
    let mut issues = Vec::new();
    if record.record_version != REPLICATION_EXECUTION_VERSION {
        issues.push(ReplicationExecutionIssue::WrongVersion {
            found: record.record_version.clone(),
        });
    }
    let protocol_digest = match replication_protocol_commitment(protocol) {
        Ok(value) if value == protocol.protocol_sha256 => value,
        _ => {
            issues.push(ReplicationExecutionIssue::InvalidProtocol);
            String::new()
        }
    };
    let registry_digest = match replication_site_registry_commitment(registry) {
        Ok(value) if value == registry.registry_sha256 => value,
        _ => {
            issues.push(ReplicationExecutionIssue::InvalidSiteRegistry);
            String::new()
        }
    };
    let package_digest = match replication_package_commitment(package) {
        Ok(value) if value == package.package_sha256 => value,
        _ => {
            issues.push(ReplicationExecutionIssue::InvalidPackage);
            String::new()
        }
    };
    if record.replication_id != protocol.replication_id
        || record.protocol_sha256 != protocol_digest
        || record.site_registry_sha256 != registry_digest
        || record.site_package_sha256 != package_digest
        || record.site_id != package.site_id
        || record.analysis_plan_sha256 != protocol.analysis_plan_sha256
    {
        issues.push(ReplicationExecutionIssue::AuthorityMismatch);
    }
    if !registry.sites.iter().any(|site| {
        site.site_id == record.site_id && site.site_status == ReplicationSiteStatus::Registered
    }) {
        issues.push(ReplicationExecutionIssue::UnknownOrInactiveSite {
            site_id: record.site_id.clone(),
        });
    }
    for (field, value) in [
        ("replication_id", record.replication_id.as_str()),
        ("site_id", record.site_id.as_str()),
        ("public_release_uri", record.public_release_uri.as_str()),
        ("completed_at_utc", record.completed_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ReplicationExecutionIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("protocol_sha256", record.protocol_sha256.as_str()),
        ("site_registry_sha256", record.site_registry_sha256.as_str()),
        ("site_package_sha256", record.site_package_sha256.as_str()),
        (
            "package_receipt_sha256",
            record.package_receipt_sha256.as_str(),
        ),
        (
            "collection_close_sha256",
            record.collection_close_sha256.as_str(),
        ),
        ("dataset_sha256", record.dataset_sha256.as_str()),
        ("analysis_plan_sha256", record.analysis_plan_sha256.as_str()),
        (
            "primary_analysis_sha256",
            record.primary_analysis_sha256.as_str(),
        ),
        (
            "independent_analysis_sha256",
            record.independent_analysis_sha256.as_str(),
        ),
        (
            "analysis_crosscheck_sha256",
            record.analysis_crosscheck_sha256.as_str(),
        ),
        (
            "execution_environment_sha256",
            record.execution_environment_sha256.as_str(),
        ),
        (
            "source_outcome_access_audit_sha256",
            record.source_outcome_access_audit_sha256.as_str(),
        ),
        ("record_sha256", record.record_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ReplicationExecutionIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    validate_deviations(protocol, record, &mut issues);
    if let Some(result) = &record.primary_result {
        validate_primary_result(protocol, result, &mut issues);
    }
    let expected = derive_site_conclusion(protocol, record);
    if record.conclusion != expected {
        issues.push(ReplicationExecutionIssue::ConclusionMismatch {
            expected,
            found: record.conclusion,
        });
    }
    match replication_execution_commitment(record) {
        Ok(digest) if digest == record.record_sha256 => {}
        Ok(_) => issues.push(ReplicationExecutionIssue::RecordDigestMismatch),
        Err(_) => issues.push(ReplicationExecutionIssue::SerializationFailed),
    }
    issues
}

fn validate_deviations(
    protocol: &FrozenReplicationProtocol,
    record: &ReplicationSiteExecutionRecord,
    issues: &mut Vec<ReplicationExecutionIssue>,
) {
    let allowed = protocol
        .allowed_deviations
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let prohibited = protocol
        .prohibited_deviations
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for deviation in &record.deviations {
        let category = deviation.category.trim().to_ascii_lowercase();
        if !seen.insert(category.clone()) {
            issues.push(ReplicationExecutionIssue::DuplicateDeviation { category });
            continue;
        }
        let conflicts_with_protocol = prohibited.contains(&category)
            || (!allowed.contains(&category)
                && protocol.replication_kind == ReplicationKind::Direct);
        if (conflicts_with_protocol || !deviation.approved_before_collection)
            && !deviation.material_to_primary_claim
        {
            issues.push(
                ReplicationExecutionIssue::ProtocolDeviationNotMarkedMaterial {
                    category: deviation.category.clone(),
                },
            );
        }
        if !is_sha256(&deviation.evidence_sha256) {
            issues.push(ReplicationExecutionIssue::InvalidDigest {
                field: format!("deviation.{}.evidence_sha256", deviation.category),
            });
        }
    }
}

fn validate_primary_result(
    protocol: &FrozenReplicationProtocol,
    result: &SitePrimaryResult,
    issues: &mut Vec<ReplicationExecutionIssue>,
) {
    if result.endpoint_id != protocol.primary_endpoint.endpoint_id {
        issues.push(ReplicationExecutionIssue::PrimaryEndpointMismatch);
    }
    for (field, valid) in [
        ("estimate", result.estimate.is_finite()),
        (
            "standard_error",
            result.standard_error.is_finite() && result.standard_error > 0.0,
        ),
        ("confidence_lower", result.confidence_lower.is_finite()),
        ("confidence_upper", result.confidence_upper.is_finite()),
        (
            "confidence_order",
            result.confidence_lower <= result.estimate
                && result.estimate <= result.confidence_upper,
        ),
        (
            "p_value",
            result.p_value.is_finite() && (0.0..=1.0).contains(&result.p_value),
        ),
    ] {
        if !valid {
            issues.push(ReplicationExecutionIssue::InvalidPrimaryResult {
                field: field.into(),
            });
        }
    }
}

fn valid_primary_numbers(result: &SitePrimaryResult) -> bool {
    result.estimate.is_finite()
        && result.standard_error.is_finite()
        && result.standard_error > 0.0
        && result.confidence_lower.is_finite()
        && result.confidence_upper.is_finite()
        && result.confidence_lower <= result.estimate
        && result.estimate <= result.confidence_upper
        && result.p_value.is_finite()
        && (0.0..=1.0).contains(&result.p_value)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replication_protocol::{REPLICATION_PROTOCOL_VERSION, ReplicationEndpointSpec};

    fn protocol() -> FrozenReplicationProtocol {
        FrozenReplicationProtocol {
            protocol_version: REPLICATION_PROTOCOL_VERSION.into(),
            replication_id: "rep".into(),
            source_study_id: "study".into(),
            source_final_release_sha256: "a".repeat(64),
            replication_kind: ReplicationKind::Direct,
            primary_endpoint: ReplicationEndpointSpec {
                endpoint_id: "primary".into(),
                outcome_scale: "difference".into(),
                favorable_direction: FavorableDirection::Higher,
                practical_margin: 0.05,
                alpha: 0.05,
                confidence_level: 0.95,
            },
            required_site_count: 2,
            minimum_independent_organizations: 2,
            participant_target_per_site: 40,
            family_target_per_site: 20,
            analysis_plan_sha256: "b".repeat(64),
            artifact_generation_plan_sha256: "c".repeat(64),
            randomization_commitment_sha256: "d".repeat(64),
            preregistration_uri: "uri".into(),
            preregistration_receipt_sha256: "e".repeat(64),
            allowed_deviations: vec!["local consent wording".into()],
            prohibited_deviations: vec!["primary endpoint".into()],
            frozen_at_utc: "now".into(),
            protocol_sha256: "f".repeat(64),
        }
    }

    fn record() -> ReplicationSiteExecutionRecord {
        ReplicationSiteExecutionRecord {
            record_version: REPLICATION_EXECUTION_VERSION.into(),
            replication_id: "rep".into(),
            site_id: "site".into(),
            protocol_sha256: "f".repeat(64),
            site_registry_sha256: "1".repeat(64),
            site_package_sha256: "2".repeat(64),
            package_receipt_sha256: "3".repeat(64),
            collection_close_sha256: "4".repeat(64),
            dataset_sha256: "5".repeat(64),
            analysis_plan_sha256: "b".repeat(64),
            primary_analysis_sha256: "6".repeat(64),
            independent_analysis_sha256: "7".repeat(64),
            analysis_crosscheck_sha256: "8".repeat(64),
            execution_environment_sha256: "9".repeat(64),
            source_outcome_access_audit_sha256: "a".repeat(64),
            deviations: Vec::new(),
            primary_result: Some(SitePrimaryResult {
                endpoint_id: "primary".into(),
                estimate: 0.10,
                standard_error: 0.02,
                confidence_lower: 0.06,
                confidence_upper: 0.14,
                p_value: 0.001,
                participant_count: 48,
                family_count: 24,
            }),
            conclusion: SiteReplicationConclusion::Inconclusive,
            all_frozen_commands_succeeded: true,
            collection_blinded_until_close: true,
            source_outcomes_withheld_until_close: true,
            public_release_uri: "uri".into(),
            completed_at_utc: "now".into(),
            record_sha256: String::new(),
        }
    }

    #[test]
    fn conclusion_is_derived_from_frozen_margin() {
        let protocol = protocol();
        let record = record();
        assert_eq!(
            derive_site_conclusion(&protocol, &record),
            SiteReplicationConclusion::SupportsReplication
        );
    }

    #[test]
    fn material_deviation_demotes_site_result() {
        let protocol = protocol();
        let mut record = record();
        record.deviations.push(ReplicationDeviation {
            category: "unexpected change".into(),
            description: "changed after collection".into(),
            approved_before_collection: false,
            material_to_primary_claim: true,
            evidence_sha256: "b".repeat(64),
        });
        assert_eq!(
            derive_site_conclusion(&protocol, &record),
            SiteReplicationConclusion::DescriptiveOnly
        );
    }
}
