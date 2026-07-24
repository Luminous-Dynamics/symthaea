// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Privacy-minimized registry for independent replication sites.
//!
//! The registry contains commitments and role identities, not raw contact data.
//! It verifies that the frozen protocol is executed by enough genuinely
//! independent organizations with local governance and reproducibility evidence.

use crate::evidence_digest::canonical_json_sha256;
use crate::replication_protocol::{FrozenReplicationProtocol, replication_protocol_commitment};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REPLICATION_SITE_REGISTRY_VERSION: &str = "symthaea-muse-replication-site-registry-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationSiteStatus {
    Registered,
    WithdrawnBeforeCollection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationSiteRecord {
    pub site_id: String,
    pub organization_id: String,
    pub country_code: String,
    pub site_status: ReplicationSiteStatus,
    pub principal_investigator_id: String,
    pub data_custodian_id: String,
    pub analyst_id: String,
    pub contact_commitment_sha256: String,
    pub conflict_of_interest_declaration: String,
    pub independent_of_source_authors: bool,
    pub governance_approval_uri: String,
    pub governance_approval_sha256: String,
    pub environment_capability_sha256: String,
    pub local_protocol_sha256: String,
    pub registration_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationSiteRegistry {
    pub registry_version: String,
    pub replication_id: String,
    pub protocol_sha256: String,
    pub sites: Vec<ReplicationSiteRecord>,
    pub frozen_at_utc: String,
    pub registry_sha256: String,
}

#[derive(Serialize)]
struct RegistryCommitment<'a> {
    registry_version: &'a str,
    replication_id: &'a str,
    protocol_sha256: &'a str,
    sites: &'a [ReplicationSiteRecord],
    frozen_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationSiteRegistryIssue {
    WrongVersion {
        found: String,
    },
    InvalidProtocol,
    ProtocolMismatch,
    EmptyField {
        site_id: Option<String>,
        field: String,
    },
    InvalidDigest {
        site_id: Option<String>,
        field: String,
    },
    DuplicateSiteId {
        site_id: String,
    },
    DuplicateOrganization {
        organization_id: String,
    },
    InsufficientRegisteredSites {
        required: u32,
        found: u32,
    },
    InsufficientIndependentOrganizations {
        required: u32,
        found: u32,
    },
    SiteNotIndependent {
        site_id: String,
    },
    EmptyConflictDeclaration {
        site_id: String,
    },
    RoleCollision {
        site_id: String,
        role_ids: Vec<String>,
    },
    SerializationFailed,
    RegistryDigestMismatch,
}

pub fn replication_site_registry_commitment(
    registry: &ReplicationSiteRegistry,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RegistryCommitment {
        registry_version: &registry.registry_version,
        replication_id: &registry.replication_id,
        protocol_sha256: &registry.protocol_sha256,
        sites: &registry.sites,
        frozen_at_utc: &registry.frozen_at_utc,
    })
}

pub fn seal_replication_site_registry(
    protocol: &FrozenReplicationProtocol,
    registry: &mut ReplicationSiteRegistry,
) -> Result<(), Vec<ReplicationSiteRegistryIssue>> {
    registry
        .sites
        .sort_by(|left, right| left.site_id.cmp(&right.site_id));
    registry.registry_sha256 = replication_site_registry_commitment(registry)
        .map_err(|_| vec![ReplicationSiteRegistryIssue::SerializationFailed])?;
    let issues = validate_replication_site_registry(protocol, registry);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn validate_replication_site_registry(
    protocol: &FrozenReplicationProtocol,
    registry: &ReplicationSiteRegistry,
) -> Vec<ReplicationSiteRegistryIssue> {
    let mut issues = Vec::new();
    if registry.registry_version != REPLICATION_SITE_REGISTRY_VERSION {
        issues.push(ReplicationSiteRegistryIssue::WrongVersion {
            found: registry.registry_version.clone(),
        });
    }
    let expected_protocol = match replication_protocol_commitment(protocol) {
        Ok(value) if value == protocol.protocol_sha256 => value,
        _ => {
            issues.push(ReplicationSiteRegistryIssue::InvalidProtocol);
            String::new()
        }
    };
    if registry.replication_id != protocol.replication_id
        || registry.protocol_sha256 != expected_protocol
    {
        issues.push(ReplicationSiteRegistryIssue::ProtocolMismatch);
    }
    for (field, value) in [
        ("replication_id", registry.replication_id.as_str()),
        ("frozen_at_utc", registry.frozen_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ReplicationSiteRegistryIssue::EmptyField {
                site_id: None,
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        ("protocol_sha256", registry.protocol_sha256.as_str()),
        ("registry_sha256", registry.registry_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ReplicationSiteRegistryIssue::InvalidDigest {
                site_id: None,
                field: field.into(),
            });
        }
    }

    let mut site_ids = BTreeSet::new();
    let mut organization_counts = BTreeMap::<&str, usize>::new();
    let mut registered_sites = 0u32;
    for site in &registry.sites {
        if !site_ids.insert(site.site_id.as_str()) {
            issues.push(ReplicationSiteRegistryIssue::DuplicateSiteId {
                site_id: site.site_id.clone(),
            });
        }
        *organization_counts
            .entry(site.organization_id.as_str())
            .or_default() += 1;
        if site.site_status == ReplicationSiteStatus::Registered {
            registered_sites += 1;
        }
        validate_site(site, &mut issues);
    }
    for (organization_id, count) in &organization_counts {
        if *count > 1 {
            issues.push(ReplicationSiteRegistryIssue::DuplicateOrganization {
                organization_id: (*organization_id).into(),
            });
        }
    }
    let independent_organizations = registry
        .sites
        .iter()
        .filter(|site| site.site_status == ReplicationSiteStatus::Registered)
        .map(|site| site.organization_id.as_str())
        .collect::<BTreeSet<_>>()
        .len() as u32;
    if registered_sites < protocol.required_site_count {
        issues.push(ReplicationSiteRegistryIssue::InsufficientRegisteredSites {
            required: protocol.required_site_count,
            found: registered_sites,
        });
    }
    if independent_organizations < protocol.minimum_independent_organizations {
        issues.push(
            ReplicationSiteRegistryIssue::InsufficientIndependentOrganizations {
                required: protocol.minimum_independent_organizations,
                found: independent_organizations,
            },
        );
    }
    match replication_site_registry_commitment(registry) {
        Ok(digest) if digest == registry.registry_sha256 => {}
        Ok(_) => issues.push(ReplicationSiteRegistryIssue::RegistryDigestMismatch),
        Err(_) => issues.push(ReplicationSiteRegistryIssue::SerializationFailed),
    }
    issues
}

fn validate_site(site: &ReplicationSiteRecord, issues: &mut Vec<ReplicationSiteRegistryIssue>) {
    for (field, value) in [
        ("site_id", site.site_id.as_str()),
        ("organization_id", site.organization_id.as_str()),
        ("country_code", site.country_code.as_str()),
        (
            "principal_investigator_id",
            site.principal_investigator_id.as_str(),
        ),
        ("data_custodian_id", site.data_custodian_id.as_str()),
        ("analyst_id", site.analyst_id.as_str()),
        (
            "governance_approval_uri",
            site.governance_approval_uri.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            issues.push(ReplicationSiteRegistryIssue::EmptyField {
                site_id: Some(site.site_id.clone()),
                field: field.into(),
            });
        }
    }
    for (field, digest) in [
        (
            "contact_commitment_sha256",
            site.contact_commitment_sha256.as_str(),
        ),
        (
            "governance_approval_sha256",
            site.governance_approval_sha256.as_str(),
        ),
        (
            "environment_capability_sha256",
            site.environment_capability_sha256.as_str(),
        ),
        ("local_protocol_sha256", site.local_protocol_sha256.as_str()),
        (
            "registration_receipt_sha256",
            site.registration_receipt_sha256.as_str(),
        ),
    ] {
        if !is_sha256(digest) {
            issues.push(ReplicationSiteRegistryIssue::InvalidDigest {
                site_id: Some(site.site_id.clone()),
                field: field.into(),
            });
        }
    }
    if !site.independent_of_source_authors {
        issues.push(ReplicationSiteRegistryIssue::SiteNotIndependent {
            site_id: site.site_id.clone(),
        });
    }
    if site.conflict_of_interest_declaration.trim().is_empty() {
        issues.push(ReplicationSiteRegistryIssue::EmptyConflictDeclaration {
            site_id: site.site_id.clone(),
        });
    }
    let roles = [
        site.principal_investigator_id.as_str(),
        site.data_custodian_id.as_str(),
        site.analyst_id.as_str(),
    ];
    if roles.iter().copied().collect::<BTreeSet<_>>().len() != roles.len() {
        issues.push(ReplicationSiteRegistryIssue::RoleCollision {
            site_id: site.site_id.clone(),
            role_ids: roles.iter().map(|value| (*value).to_string()).collect(),
        });
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replication_protocol::{
        FavorableDirection, REPLICATION_PROTOCOL_VERSION, ReplicationEndpointSpec, ReplicationKind,
    };

    fn protocol() -> FrozenReplicationProtocol {
        let mut protocol = FrozenReplicationProtocol {
            protocol_version: REPLICATION_PROTOCOL_VERSION.into(),
            replication_id: "rep-1".into(),
            source_study_id: "study".into(),
            source_final_release_sha256: "a".repeat(64),
            replication_kind: ReplicationKind::Direct,
            primary_endpoint: ReplicationEndpointSpec {
                endpoint_id: "primary".into(),
                outcome_scale: "ordinal".into(),
                favorable_direction: FavorableDirection::Higher,
                practical_margin: 0.05,
                alpha: 0.05,
                confidence_level: 0.95,
            },
            required_site_count: 2,
            minimum_independent_organizations: 2,
            participant_target_per_site: 48,
            family_target_per_site: 24,
            analysis_plan_sha256: "b".repeat(64),
            artifact_generation_plan_sha256: "c".repeat(64),
            randomization_commitment_sha256: "d".repeat(64),
            preregistration_uri: "https://example.invalid/pre".into(),
            preregistration_receipt_sha256: "e".repeat(64),
            allowed_deviations: Vec::new(),
            prohibited_deviations: Vec::new(),
            frozen_at_utc: "now".into(),
            protocol_sha256: String::new(),
        };
        protocol.protocol_sha256 =
            crate::replication_protocol::replication_protocol_commitment(&protocol).unwrap();
        protocol
    }

    fn site(id: &str, organization: &str, fill: char) -> ReplicationSiteRecord {
        ReplicationSiteRecord {
            site_id: id.into(),
            organization_id: organization.into(),
            country_code: "ZA".into(),
            site_status: ReplicationSiteStatus::Registered,
            principal_investigator_id: format!("{id}-pi"),
            data_custodian_id: format!("{id}-custodian"),
            analyst_id: format!("{id}-analyst"),
            contact_commitment_sha256: fill.to_string().repeat(64),
            conflict_of_interest_declaration: "none".into(),
            independent_of_source_authors: true,
            governance_approval_uri: "https://example.invalid/approval".into(),
            governance_approval_sha256: fill.to_string().repeat(64),
            environment_capability_sha256: fill.to_string().repeat(64),
            local_protocol_sha256: fill.to_string().repeat(64),
            registration_receipt_sha256: fill.to_string().repeat(64),
        }
    }

    #[test]
    fn distinct_sites_satisfy_registry() {
        let protocol = protocol();
        let mut registry = ReplicationSiteRegistry {
            registry_version: REPLICATION_SITE_REGISTRY_VERSION.into(),
            replication_id: protocol.replication_id.clone(),
            protocol_sha256: protocol.protocol_sha256.clone(),
            sites: vec![site("site-a", "org-a", '1'), site("site-b", "org-b", '2')],
            frozen_at_utc: "now".into(),
            registry_sha256: String::new(),
        };
        seal_replication_site_registry(&protocol, &mut registry).unwrap();
        assert!(validate_replication_site_registry(&protocol, &registry).is_empty());
    }

    #[test]
    fn duplicate_organization_is_rejected() {
        let protocol = protocol();
        let mut registry = ReplicationSiteRegistry {
            registry_version: REPLICATION_SITE_REGISTRY_VERSION.into(),
            replication_id: protocol.replication_id.clone(),
            protocol_sha256: protocol.protocol_sha256.clone(),
            sites: vec![site("site-a", "org-a", '1'), site("site-b", "org-a", '2')],
            frozen_at_utc: "now".into(),
            registry_sha256: String::new(),
        };
        registry.registry_sha256 = replication_site_registry_commitment(&registry).unwrap();
        assert!(
            validate_replication_site_registry(&protocol, &registry)
                .iter()
                .any(|issue| matches!(
                    issue,
                    ReplicationSiteRegistryIssue::DuplicateOrganization { .. }
                ))
        );
    }
}
