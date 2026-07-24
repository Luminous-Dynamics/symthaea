// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root commitment for independent replication and long-term stewardship.
//!
//! The V13 release links the immutable V12 publication, prospective replication
//! authorities, every site execution, cross-site synthesis, durable archive,
//! distributed governance, and evidence-derived research-release promotion.

use crate::confirmatory_final_release::{
    ConfirmatoryFinalReleaseBundle, confirmatory_final_release_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::replication_execution::{
    ReplicationSiteExecutionRecord, replication_execution_commitment,
    validate_replication_execution,
};
use crate::replication_orchestration::{
    ReplicationLifecyclePhase, ReplicationOrchestrationLog, replication_orchestration_commitment,
    validate_replication_orchestration,
};
use crate::replication_package::{
    ReplicationSitePackage, replication_package_commitment, validate_replication_package,
};
use crate::replication_protocol::{
    FrozenReplicationProtocol, replication_protocol_commitment, validate_replication_protocol,
};
use crate::replication_site_registry::{
    ReplicationSiteRegistry, ReplicationSiteStatus, replication_site_registry_commitment,
    validate_replication_site_registry,
};
use crate::replication_synthesis::{
    ReplicationSynthesisConclusion, ReplicationSynthesisRecord, replication_synthesis_commitment,
    validate_replication_synthesis,
};
use crate::research_archive::{ResearchArchiveManifest, research_archive_commitment};
use crate::research_release_promotion::{
    ResearchReleasePromotionRecord, ResearchReleaseStage, research_release_promotion_commitment,
};
use crate::stewardship_governance::{
    ResearchStewardshipCharter, stewardship_charter_commitment, validate_stewardship_charter,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const STEWARDSHIP_RELEASE_VERSION: &str = "symthaea-muse-stewardship-release-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StewardshipReleaseBundle {
    pub release_version: String,
    pub stewardship_id: String,
    pub source_final_release_sha256: String,
    pub replication_protocol_sha256: String,
    pub site_registry_sha256: String,
    pub site_packages_root_sha256: String,
    pub site_executions_root_sha256: String,
    pub replication_synthesis_sha256: String,
    pub replication_orchestration_sha256: String,
    pub stewardship_charter_sha256: String,
    pub research_archive_sha256: String,
    pub release_promotion_sha256: String,
    pub revision_governance_policy_sha256: String,
    pub security_review_sha256: String,
    pub source_revision: String,
    pub workspace_tree_sha256: String,
    pub public_release_uri: String,
    pub released_at_utc: String,
    pub bundle_sha256: String,
}

#[derive(Serialize)]
struct StewardshipReleaseCommitment<'a> {
    release_version: &'a str,
    stewardship_id: &'a str,
    source_final_release_sha256: &'a str,
    replication_protocol_sha256: &'a str,
    site_registry_sha256: &'a str,
    site_packages_root_sha256: &'a str,
    site_executions_root_sha256: &'a str,
    replication_synthesis_sha256: &'a str,
    replication_orchestration_sha256: &'a str,
    stewardship_charter_sha256: &'a str,
    research_archive_sha256: &'a str,
    release_promotion_sha256: &'a str,
    revision_governance_policy_sha256: &'a str,
    security_review_sha256: &'a str,
    source_revision: &'a str,
    workspace_tree_sha256: &'a str,
    public_release_uri: &'a str,
    released_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StewardshipReleaseIssue {
    WrongVersion { found: String },
    InvalidSourceRelease,
    InvalidReplicationProtocol,
    InvalidSiteRegistry,
    InvalidSitePackage { site_id: String },
    InvalidSiteExecution { site_id: String },
    InvalidReplicationSynthesis,
    InvalidReplicationOrchestration,
    InvalidStewardshipCharter,
    InvalidResearchArchive,
    InvalidReleasePromotion,
    AuthorityMismatch { field: String },
    MissingPackage { site_id: String },
    MissingExecution { site_id: String },
    DuplicatePackage { site_id: String },
    DuplicateExecution { site_id: String },
    ReplicationNotEstablished,
    OrchestrationNotReleased,
    ReleaseNotPromoted,
    EmptyField { field: String },
    InvalidDigest { field: String },
    DerivedRootMismatch { field: String },
    SerializationFailed,
    BundleDigestMismatch,
}

pub fn site_packages_root_commitment(
    packages: &[ReplicationSitePackage],
) -> Result<String, serde_json::Error> {
    let mut bindings = packages
        .iter()
        .map(|package| (package.site_id.clone(), package.package_sha256.clone()))
        .collect::<Vec<_>>();
    bindings.sort();
    canonical_json_sha256(&bindings)
}

pub fn site_executions_root_commitment(
    records: &[ReplicationSiteExecutionRecord],
) -> Result<String, serde_json::Error> {
    let mut bindings = records
        .iter()
        .map(|record| (record.site_id.clone(), record.record_sha256.clone()))
        .collect::<Vec<_>>();
    bindings.sort();
    canonical_json_sha256(&bindings)
}

pub fn stewardship_release_commitment(
    bundle: &StewardshipReleaseBundle,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&StewardshipReleaseCommitment {
        release_version: &bundle.release_version,
        stewardship_id: &bundle.stewardship_id,
        source_final_release_sha256: &bundle.source_final_release_sha256,
        replication_protocol_sha256: &bundle.replication_protocol_sha256,
        site_registry_sha256: &bundle.site_registry_sha256,
        site_packages_root_sha256: &bundle.site_packages_root_sha256,
        site_executions_root_sha256: &bundle.site_executions_root_sha256,
        replication_synthesis_sha256: &bundle.replication_synthesis_sha256,
        replication_orchestration_sha256: &bundle.replication_orchestration_sha256,
        stewardship_charter_sha256: &bundle.stewardship_charter_sha256,
        research_archive_sha256: &bundle.research_archive_sha256,
        release_promotion_sha256: &bundle.release_promotion_sha256,
        revision_governance_policy_sha256: &bundle.revision_governance_policy_sha256,
        security_review_sha256: &bundle.security_review_sha256,
        source_revision: &bundle.source_revision,
        workspace_tree_sha256: &bundle.workspace_tree_sha256,
        public_release_uri: &bundle.public_release_uri,
        released_at_utc: &bundle.released_at_utc,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_stewardship_release(
    source_release: &ConfirmatoryFinalReleaseBundle,
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    packages: &[ReplicationSitePackage],
    executions: &[ReplicationSiteExecutionRecord],
    synthesis: &ReplicationSynthesisRecord,
    orchestration: &ReplicationOrchestrationLog,
    charter: &ResearchStewardshipCharter,
    archive: &ResearchArchiveManifest,
    promotion: &ResearchReleasePromotionRecord,
    revision_governance_policy_sha256: String,
    security_review_sha256: String,
    source_revision: String,
    workspace_tree_sha256: String,
    public_release_uri: String,
    released_at_utc: String,
) -> Result<StewardshipReleaseBundle, Vec<StewardshipReleaseIssue>> {
    let mut bundle = StewardshipReleaseBundle {
        release_version: STEWARDSHIP_RELEASE_VERSION.into(),
        stewardship_id: charter.stewardship_id.clone(),
        source_final_release_sha256: source_release.bundle_sha256.clone(),
        replication_protocol_sha256: protocol.protocol_sha256.clone(),
        site_registry_sha256: registry.registry_sha256.clone(),
        site_packages_root_sha256: site_packages_root_commitment(packages)
            .map_err(|_| vec![StewardshipReleaseIssue::SerializationFailed])?,
        site_executions_root_sha256: site_executions_root_commitment(executions)
            .map_err(|_| vec![StewardshipReleaseIssue::SerializationFailed])?,
        replication_synthesis_sha256: synthesis.synthesis_sha256.clone(),
        replication_orchestration_sha256: orchestration.log_sha256.clone(),
        stewardship_charter_sha256: charter.charter_sha256.clone(),
        research_archive_sha256: archive.archive_sha256.clone(),
        release_promotion_sha256: promotion.promotion_sha256.clone(),
        revision_governance_policy_sha256,
        security_review_sha256,
        source_revision,
        workspace_tree_sha256,
        public_release_uri,
        released_at_utc,
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = stewardship_release_commitment(&bundle)
        .map_err(|_| vec![StewardshipReleaseIssue::SerializationFailed])?;
    let issues = validate_stewardship_release(
        source_release,
        protocol,
        registry,
        packages,
        executions,
        synthesis,
        orchestration,
        charter,
        archive,
        promotion,
        &bundle,
    );
    if issues.is_empty() {
        Ok(bundle)
    } else {
        Err(issues)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_stewardship_release(
    source_release: &ConfirmatoryFinalReleaseBundle,
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
    packages: &[ReplicationSitePackage],
    executions: &[ReplicationSiteExecutionRecord],
    synthesis: &ReplicationSynthesisRecord,
    orchestration: &ReplicationOrchestrationLog,
    charter: &ResearchStewardshipCharter,
    archive: &ResearchArchiveManifest,
    promotion: &ResearchReleasePromotionRecord,
    bundle: &StewardshipReleaseBundle,
) -> Vec<StewardshipReleaseIssue> {
    let mut issues = Vec::new();
    if bundle.release_version != STEWARDSHIP_RELEASE_VERSION {
        issues.push(StewardshipReleaseIssue::WrongVersion {
            found: bundle.release_version.clone(),
        });
    }
    let source_digest = checked_digest(
        confirmatory_final_release_commitment(source_release),
        &source_release.bundle_sha256,
        StewardshipReleaseIssue::InvalidSourceRelease,
        &mut issues,
    );
    let protocol_digest = checked_digest(
        replication_protocol_commitment(protocol),
        &protocol.protocol_sha256,
        StewardshipReleaseIssue::InvalidReplicationProtocol,
        &mut issues,
    );
    let registry_digest = checked_digest(
        replication_site_registry_commitment(registry),
        &registry.registry_sha256,
        StewardshipReleaseIssue::InvalidSiteRegistry,
        &mut issues,
    );
    let synthesis_digest = checked_digest(
        replication_synthesis_commitment(synthesis),
        &synthesis.synthesis_sha256,
        StewardshipReleaseIssue::InvalidReplicationSynthesis,
        &mut issues,
    );
    let orchestration_digest = checked_digest(
        replication_orchestration_commitment(orchestration),
        &orchestration.log_sha256,
        StewardshipReleaseIssue::InvalidReplicationOrchestration,
        &mut issues,
    );
    let charter_digest = checked_digest(
        stewardship_charter_commitment(charter),
        &charter.charter_sha256,
        StewardshipReleaseIssue::InvalidStewardshipCharter,
        &mut issues,
    );
    let archive_digest = checked_digest(
        research_archive_commitment(archive),
        &archive.archive_sha256,
        StewardshipReleaseIssue::InvalidResearchArchive,
        &mut issues,
    );
    let promotion_digest = checked_digest(
        research_release_promotion_commitment(promotion),
        &promotion.promotion_sha256,
        StewardshipReleaseIssue::InvalidReleasePromotion,
        &mut issues,
    );

    if !validate_replication_protocol(source_release, protocol).is_empty() {
        issues.push(StewardshipReleaseIssue::InvalidReplicationProtocol);
    }
    if !validate_replication_site_registry(protocol, registry).is_empty() {
        issues.push(StewardshipReleaseIssue::InvalidSiteRegistry);
    }
    if !validate_replication_synthesis(protocol, registry, executions, synthesis).is_empty() {
        issues.push(StewardshipReleaseIssue::InvalidReplicationSynthesis);
    }
    if !validate_stewardship_charter(source_release, charter).is_empty() {
        issues.push(StewardshipReleaseIssue::InvalidStewardshipCharter);
    }

    let package_root = site_packages_root_commitment(packages).unwrap_or_default();
    let execution_root = site_executions_root_commitment(executions).unwrap_or_default();
    validate_sites(registry, packages, executions, &mut issues);
    for package in packages {
        let commitment_valid = replication_package_commitment(package)
            .is_ok_and(|digest| digest == package.package_sha256);
        if !commitment_valid
            || !validate_replication_package(protocol, registry, package).is_empty()
        {
            issues.push(StewardshipReleaseIssue::InvalidSitePackage {
                site_id: package.site_id.clone(),
            });
        }
    }
    for execution in executions {
        let package = packages
            .iter()
            .find(|package| package.site_id == execution.site_id);
        let commitment_valid = replication_execution_commitment(execution)
            .is_ok_and(|digest| digest == execution.record_sha256);
        let semantic_valid = package.is_some_and(|package| {
            validate_replication_execution(protocol, registry, package, execution).is_empty()
        });
        if !commitment_valid || !semantic_valid {
            issues.push(StewardshipReleaseIssue::InvalidSiteExecution {
                site_id: execution.site_id.clone(),
            });
        }
    }
    for (field, expected, found) in [
        (
            "source_final_release_sha256",
            source_digest.as_str(),
            bundle.source_final_release_sha256.as_str(),
        ),
        (
            "replication_protocol_sha256",
            protocol_digest.as_str(),
            bundle.replication_protocol_sha256.as_str(),
        ),
        (
            "site_registry_sha256",
            registry_digest.as_str(),
            bundle.site_registry_sha256.as_str(),
        ),
        (
            "replication_synthesis_sha256",
            synthesis_digest.as_str(),
            bundle.replication_synthesis_sha256.as_str(),
        ),
        (
            "replication_orchestration_sha256",
            orchestration_digest.as_str(),
            bundle.replication_orchestration_sha256.as_str(),
        ),
        (
            "stewardship_charter_sha256",
            charter_digest.as_str(),
            bundle.stewardship_charter_sha256.as_str(),
        ),
        (
            "research_archive_sha256",
            archive_digest.as_str(),
            bundle.research_archive_sha256.as_str(),
        ),
        (
            "release_promotion_sha256",
            promotion_digest.as_str(),
            bundle.release_promotion_sha256.as_str(),
        ),
    ] {
        if expected != found {
            issues.push(StewardshipReleaseIssue::AuthorityMismatch {
                field: field.into(),
            });
        }
    }
    if bundle.site_packages_root_sha256 != package_root {
        issues.push(StewardshipReleaseIssue::DerivedRootMismatch {
            field: "site_packages_root_sha256".into(),
        });
    }
    if bundle.site_executions_root_sha256 != execution_root {
        issues.push(StewardshipReleaseIssue::DerivedRootMismatch {
            field: "site_executions_root_sha256".into(),
        });
    }
    if protocol.source_final_release_sha256 != source_digest
        || registry.protocol_sha256 != protocol_digest
        || synthesis.protocol_sha256 != protocol_digest
        || synthesis.site_registry_sha256 != registry_digest
        || charter.source_final_release_sha256 != source_digest
        || archive.stewardship_id != charter.stewardship_id
        || archive.authority_root_sha256 != synthesis_digest
        || promotion.source_final_release_sha256 != source_digest
        || promotion.replication_synthesis_sha256 != synthesis_digest
        || promotion.research_archive_sha256 != archive_digest
        || promotion.stewardship_charter_sha256 != charter_digest
        || bundle.stewardship_id != charter.stewardship_id
    {
        issues.push(StewardshipReleaseIssue::AuthorityMismatch {
            field: "cross-authority lineage".into(),
        });
    }
    if synthesis.conclusion != ReplicationSynthesisConclusion::IndependentlyReplicated {
        issues.push(StewardshipReleaseIssue::ReplicationNotEstablished);
    }
    if !validate_replication_orchestration(orchestration).is_empty()
        || orchestration.current_phase != ReplicationLifecyclePhase::StewardshipReleased
    {
        issues.push(StewardshipReleaseIssue::OrchestrationNotReleased);
    }
    if !promotion.promoted || promotion.target_stage != ResearchReleaseStage::StableResearchRelease
    {
        issues.push(StewardshipReleaseIssue::ReleaseNotPromoted);
    }
    for (field, value) in [
        ("stewardship_id", bundle.stewardship_id.as_str()),
        ("source_revision", bundle.source_revision.as_str()),
        ("public_release_uri", bundle.public_release_uri.as_str()),
        ("released_at_utc", bundle.released_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(StewardshipReleaseIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "source_final_release_sha256",
            bundle.source_final_release_sha256.as_str(),
        ),
        (
            "replication_protocol_sha256",
            bundle.replication_protocol_sha256.as_str(),
        ),
        ("site_registry_sha256", bundle.site_registry_sha256.as_str()),
        (
            "site_packages_root_sha256",
            bundle.site_packages_root_sha256.as_str(),
        ),
        (
            "site_executions_root_sha256",
            bundle.site_executions_root_sha256.as_str(),
        ),
        (
            "replication_synthesis_sha256",
            bundle.replication_synthesis_sha256.as_str(),
        ),
        (
            "replication_orchestration_sha256",
            bundle.replication_orchestration_sha256.as_str(),
        ),
        (
            "stewardship_charter_sha256",
            bundle.stewardship_charter_sha256.as_str(),
        ),
        (
            "research_archive_sha256",
            bundle.research_archive_sha256.as_str(),
        ),
        (
            "release_promotion_sha256",
            bundle.release_promotion_sha256.as_str(),
        ),
        (
            "revision_governance_policy_sha256",
            bundle.revision_governance_policy_sha256.as_str(),
        ),
        (
            "security_review_sha256",
            bundle.security_review_sha256.as_str(),
        ),
        (
            "workspace_tree_sha256",
            bundle.workspace_tree_sha256.as_str(),
        ),
        ("bundle_sha256", bundle.bundle_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(StewardshipReleaseIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match stewardship_release_commitment(bundle) {
        Ok(digest) if digest == bundle.bundle_sha256 => {}
        Ok(_) => issues.push(StewardshipReleaseIssue::BundleDigestMismatch),
        Err(_) => issues.push(StewardshipReleaseIssue::SerializationFailed),
    }
    issues
}

fn validate_sites(
    registry: &ReplicationSiteRegistry,
    packages: &[ReplicationSitePackage],
    executions: &[ReplicationSiteExecutionRecord],
    issues: &mut Vec<StewardshipReleaseIssue>,
) {
    let mut package_counts = BTreeMap::<&str, usize>::new();
    for package in packages {
        *package_counts.entry(package.site_id.as_str()).or_default() += 1;
    }
    let mut execution_counts = BTreeMap::<&str, usize>::new();
    for execution in executions {
        *execution_counts
            .entry(execution.site_id.as_str())
            .or_default() += 1;
    }
    let active_sites = registry
        .sites
        .iter()
        .filter(|site| site.site_status == ReplicationSiteStatus::Registered)
        .map(|site| site.site_id.as_str())
        .collect::<BTreeSet<_>>();
    for site_id in active_sites {
        match package_counts.get(site_id).copied().unwrap_or_default() {
            0 => issues.push(StewardshipReleaseIssue::MissingPackage {
                site_id: site_id.into(),
            }),
            1 => {}
            _ => issues.push(StewardshipReleaseIssue::DuplicatePackage {
                site_id: site_id.into(),
            }),
        }
        match execution_counts.get(site_id).copied().unwrap_or_default() {
            0 => issues.push(StewardshipReleaseIssue::MissingExecution {
                site_id: site_id.into(),
            }),
            1 => {}
            _ => issues.push(StewardshipReleaseIssue::DuplicateExecution {
                site_id: site_id.into(),
            }),
        }
    }
}

fn checked_digest(
    calculated: Result<String, serde_json::Error>,
    stored: &str,
    issue: StewardshipReleaseIssue,
    issues: &mut Vec<StewardshipReleaseIssue>,
) -> String {
    match calculated {
        Ok(digest) if digest == stored => digest,
        _ => {
            issues.push(issue);
            String::new()
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_are_order_independent() {
        let package = |site: &str, digest: char| ReplicationSitePackage {
            package_version: "v".into(),
            replication_id: "rep".into(),
            site_id: site.into(),
            protocol_sha256: "a".repeat(64),
            site_registry_sha256: "b".repeat(64),
            entries: Vec::new(),
            issued_by: "issuer".into(),
            issued_at_utc: "now".into(),
            receipt_required: true,
            package_sha256: digest.to_string().repeat(64),
        };
        let left = vec![package("b", '2'), package("a", '1')];
        let right = vec![package("a", '1'), package("b", '2')];
        assert_eq!(
            site_packages_root_commitment(&left).unwrap(),
            site_packages_root_commitment(&right).unwrap()
        );
    }
}
