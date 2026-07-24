// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen authority for independent replication of a published cognition study.
//!
//! Replication is a new study, not a reinterpretation of the original result.
//! The protocol is bound to one immutable V12 final release and freezes the
//! primary endpoint, site count, sample targets, randomization authority, and
//! permitted deviations before any replication participant is enrolled.

use crate::confirmatory_final_release::{
    ConfirmatoryFinalReleaseBundle, confirmatory_final_release_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REPLICATION_PROTOCOL_VERSION: &str = "symthaea-muse-replication-protocol-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationKind {
    Direct,
    RegisteredExtension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FavorableDirection {
    Higher,
    Lower,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicationEndpointSpec {
    pub endpoint_id: String,
    pub outcome_scale: String,
    pub favorable_direction: FavorableDirection,
    pub practical_margin: f64,
    pub alpha: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenReplicationProtocol {
    pub protocol_version: String,
    pub replication_id: String,
    pub source_study_id: String,
    pub source_final_release_sha256: String,
    pub replication_kind: ReplicationKind,
    pub primary_endpoint: ReplicationEndpointSpec,
    pub required_site_count: u32,
    pub minimum_independent_organizations: u32,
    pub participant_target_per_site: u32,
    pub family_target_per_site: u32,
    pub analysis_plan_sha256: String,
    pub artifact_generation_plan_sha256: String,
    pub randomization_commitment_sha256: String,
    pub preregistration_uri: String,
    pub preregistration_receipt_sha256: String,
    pub allowed_deviations: Vec<String>,
    pub prohibited_deviations: Vec<String>,
    pub frozen_at_utc: String,
    pub protocol_sha256: String,
}

#[derive(Serialize)]
struct ProtocolCommitment<'a> {
    protocol_version: &'a str,
    replication_id: &'a str,
    source_study_id: &'a str,
    source_final_release_sha256: &'a str,
    replication_kind: ReplicationKind,
    primary_endpoint: &'a ReplicationEndpointSpec,
    required_site_count: u32,
    minimum_independent_organizations: u32,
    participant_target_per_site: u32,
    family_target_per_site: u32,
    analysis_plan_sha256: &'a str,
    artifact_generation_plan_sha256: &'a str,
    randomization_commitment_sha256: &'a str,
    preregistration_uri: &'a str,
    preregistration_receipt_sha256: &'a str,
    allowed_deviations: &'a [String],
    prohibited_deviations: &'a [String],
    frozen_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationProtocolIssue {
    WrongVersion { found: String },
    SourceReleaseInvalid,
    SourceReleaseMismatch,
    EmptyField { field: String },
    InvalidDigest { field: String },
    InvalidNumericField { field: String },
    TooFewSites,
    TooFewIndependentOrganizations,
    OrganizationRequirementExceedsSites,
    DuplicateDeviation { value: String },
    DeviationConflict { value: String },
    DirectReplicationAllowsCoreDeviation { value: String },
    SerializationFailed,
    ProtocolDigestMismatch,
}

pub fn replication_protocol_commitment(
    protocol: &FrozenReplicationProtocol,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ProtocolCommitment {
        protocol_version: &protocol.protocol_version,
        replication_id: &protocol.replication_id,
        source_study_id: &protocol.source_study_id,
        source_final_release_sha256: &protocol.source_final_release_sha256,
        replication_kind: protocol.replication_kind,
        primary_endpoint: &protocol.primary_endpoint,
        required_site_count: protocol.required_site_count,
        minimum_independent_organizations: protocol.minimum_independent_organizations,
        participant_target_per_site: protocol.participant_target_per_site,
        family_target_per_site: protocol.family_target_per_site,
        analysis_plan_sha256: &protocol.analysis_plan_sha256,
        artifact_generation_plan_sha256: &protocol.artifact_generation_plan_sha256,
        randomization_commitment_sha256: &protocol.randomization_commitment_sha256,
        preregistration_uri: &protocol.preregistration_uri,
        preregistration_receipt_sha256: &protocol.preregistration_receipt_sha256,
        allowed_deviations: &protocol.allowed_deviations,
        prohibited_deviations: &protocol.prohibited_deviations,
        frozen_at_utc: &protocol.frozen_at_utc,
    })
}

pub fn seal_replication_protocol(
    source_release: &ConfirmatoryFinalReleaseBundle,
    protocol: &mut FrozenReplicationProtocol,
) -> Result<(), Vec<ReplicationProtocolIssue>> {
    protocol.allowed_deviations.sort();
    protocol.allowed_deviations.dedup();
    protocol.prohibited_deviations.sort();
    protocol.prohibited_deviations.dedup();
    protocol.protocol_sha256 = replication_protocol_commitment(protocol)
        .map_err(|_| vec![ReplicationProtocolIssue::SerializationFailed])?;
    let issues = validate_replication_protocol(source_release, protocol);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_replication_protocol(
    source_release: &ConfirmatoryFinalReleaseBundle,
    protocol: &FrozenReplicationProtocol,
) -> Vec<ReplicationProtocolIssue> {
    let mut issues = Vec::new();
    if protocol.protocol_version != REPLICATION_PROTOCOL_VERSION {
        issues.push(ReplicationProtocolIssue::WrongVersion {
            found: protocol.protocol_version.clone(),
        });
    }
    match confirmatory_final_release_commitment(source_release) {
        Ok(digest) if digest == source_release.bundle_sha256 => {}
        _ => issues.push(ReplicationProtocolIssue::SourceReleaseInvalid),
    }
    if protocol.source_study_id != source_release.study_id
        || protocol.source_final_release_sha256 != source_release.bundle_sha256
    {
        issues.push(ReplicationProtocolIssue::SourceReleaseMismatch);
    }
    for (field, value) in [
        ("replication_id", protocol.replication_id.as_str()),
        ("source_study_id", protocol.source_study_id.as_str()),
        (
            "primary_endpoint.endpoint_id",
            protocol.primary_endpoint.endpoint_id.as_str(),
        ),
        (
            "primary_endpoint.outcome_scale",
            protocol.primary_endpoint.outcome_scale.as_str(),
        ),
        ("preregistration_uri", protocol.preregistration_uri.as_str()),
        ("frozen_at_utc", protocol.frozen_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ReplicationProtocolIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "source_final_release_sha256",
            protocol.source_final_release_sha256.as_str(),
        ),
        (
            "analysis_plan_sha256",
            protocol.analysis_plan_sha256.as_str(),
        ),
        (
            "artifact_generation_plan_sha256",
            protocol.artifact_generation_plan_sha256.as_str(),
        ),
        (
            "randomization_commitment_sha256",
            protocol.randomization_commitment_sha256.as_str(),
        ),
        (
            "preregistration_receipt_sha256",
            protocol.preregistration_receipt_sha256.as_str(),
        ),
        ("protocol_sha256", protocol.protocol_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ReplicationProtocolIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, valid) in [
        (
            "primary_endpoint.practical_margin",
            protocol.primary_endpoint.practical_margin.is_finite()
                && protocol.primary_endpoint.practical_margin >= 0.0,
        ),
        (
            "primary_endpoint.alpha",
            protocol.primary_endpoint.alpha.is_finite()
                && protocol.primary_endpoint.alpha > 0.0
                && protocol.primary_endpoint.alpha <= 0.05,
        ),
        (
            "primary_endpoint.confidence_level",
            protocol.primary_endpoint.confidence_level.is_finite()
                && (protocol.primary_endpoint.confidence_level - 0.95).abs() <= f64::EPSILON,
        ),
        (
            "participant_target_per_site",
            protocol.participant_target_per_site > 0,
        ),
        (
            "family_target_per_site",
            protocol.family_target_per_site > 0,
        ),
    ] {
        if !valid {
            issues.push(ReplicationProtocolIssue::InvalidNumericField {
                field: field.into(),
            });
        }
    }
    if protocol.required_site_count < 2 {
        issues.push(ReplicationProtocolIssue::TooFewSites);
    }
    if protocol.minimum_independent_organizations < 2 {
        issues.push(ReplicationProtocolIssue::TooFewIndependentOrganizations);
    }
    if protocol.minimum_independent_organizations > protocol.required_site_count {
        issues.push(ReplicationProtocolIssue::OrganizationRequirementExceedsSites);
    }
    validate_deviations(protocol, &mut issues);
    match replication_protocol_commitment(protocol) {
        Ok(digest) if digest == protocol.protocol_sha256 => {}
        Ok(_) => issues.push(ReplicationProtocolIssue::ProtocolDigestMismatch),
        Err(_) => issues.push(ReplicationProtocolIssue::SerializationFailed),
    }
    issues
}

fn validate_deviations(
    protocol: &FrozenReplicationProtocol,
    issues: &mut Vec<ReplicationProtocolIssue>,
) {
    let mut allowed = BTreeSet::new();
    for value in &protocol.allowed_deviations {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            issues.push(ReplicationProtocolIssue::EmptyField {
                field: "allowed_deviations".into(),
            });
        } else if !allowed.insert(normalized.clone()) {
            issues.push(ReplicationProtocolIssue::DuplicateDeviation { value: normalized });
        }
    }
    let mut prohibited = BTreeSet::new();
    for value in &protocol.prohibited_deviations {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            issues.push(ReplicationProtocolIssue::EmptyField {
                field: "prohibited_deviations".into(),
            });
        } else if !prohibited.insert(normalized.clone()) {
            issues.push(ReplicationProtocolIssue::DuplicateDeviation {
                value: normalized.clone(),
            });
        }
        if allowed.contains(&normalized) {
            issues.push(ReplicationProtocolIssue::DeviationConflict { value: normalized });
        }
    }
    if protocol.replication_kind == ReplicationKind::Direct {
        for core in [
            "primary endpoint",
            "policy arms",
            "analysis plan",
            "blinding",
        ] {
            if allowed.iter().any(|value| value.contains(core)) {
                issues.push(
                    ReplicationProtocolIssue::DirectReplicationAllowsCoreDeviation {
                        value: core.into(),
                    },
                );
            }
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_release() -> ConfirmatoryFinalReleaseBundle {
        let mut release = ConfirmatoryFinalReleaseBundle {
            release_version: "symthaea-muse-confirmatory-final-release-v1".into(),
            study_id: "source-study".into(),
            readiness_release_sha256: "1".repeat(64),
            collection_protocol_sha256: "2".repeat(64),
            collection_close_sha256: "3".repeat(64),
            unblinding_receipt_sha256: "4".repeat(64),
            analysis_execution_sha256: "5".repeat(64),
            publication_record_sha256: "6".repeat(64),
            post_publication_audit_sha256: "7".repeat(64),
            study_release_bundle_sha256: "8".repeat(64),
            orchestration_log_sha256: "9".repeat(64),
            source_revision: "rev".into(),
            workspace_tree_sha256: "a".repeat(64),
            execution_environment_sha256: "b".repeat(64),
            public_release_uri: "https://example.invalid/release".into(),
            released_at_utc: "2026-07-14T00:00:00Z".into(),
            bundle_sha256: String::new(),
        };
        release.bundle_sha256 = confirmatory_final_release_commitment(&release).unwrap();
        release
    }

    fn protocol(release: &ConfirmatoryFinalReleaseBundle) -> FrozenReplicationProtocol {
        FrozenReplicationProtocol {
            protocol_version: REPLICATION_PROTOCOL_VERSION.into(),
            replication_id: "replication-1".into(),
            source_study_id: release.study_id.clone(),
            source_final_release_sha256: release.bundle_sha256.clone(),
            replication_kind: ReplicationKind::Direct,
            primary_endpoint: ReplicationEndpointSpec {
                endpoint_id: "preference-rank".into(),
                outcome_scale: "paired ordinal preference".into(),
                favorable_direction: FavorableDirection::Higher,
                practical_margin: 0.05,
                alpha: 0.05,
                confidence_level: 0.95,
            },
            required_site_count: 2,
            minimum_independent_organizations: 2,
            participant_target_per_site: 48,
            family_target_per_site: 24,
            analysis_plan_sha256: "c".repeat(64),
            artifact_generation_plan_sha256: "d".repeat(64),
            randomization_commitment_sha256: "e".repeat(64),
            preregistration_uri: "https://example.invalid/preregistration".into(),
            preregistration_receipt_sha256: "f".repeat(64),
            allowed_deviations: vec!["local consent wording".into()],
            prohibited_deviations: vec!["primary endpoint".into(), "policy arms".into()],
            frozen_at_utc: "2026-07-14T00:00:00Z".into(),
            protocol_sha256: String::new(),
        }
    }

    #[test]
    fn valid_direct_replication_protocol_seals() {
        let release = source_release();
        let mut protocol = protocol(&release);
        seal_replication_protocol(&release, &mut protocol).unwrap();
        assert!(validate_replication_protocol(&release, &protocol).is_empty());
    }

    #[test]
    fn direct_replication_cannot_change_primary_endpoint() {
        let release = source_release();
        let mut protocol = protocol(&release);
        protocol
            .allowed_deviations
            .push("primary endpoint scoring".into());
        protocol.protocol_sha256 = replication_protocol_commitment(&protocol).unwrap();
        let issues = validate_replication_protocol(&release, &protocol);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ReplicationProtocolIssue::DirectReplicationAllowsCoreDeviation { .. }
        )));
    }
}
