// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Digest-bound configuration diffs and conservative revalidation-impact analysis.
//!
//! A changed configuration is a new engineering baseline. This module identifies
//! changed artifacts and domains that need reassessment; it never certifies safety
//! and never asserts that prior evidence remains applicable.

use crate::modular_interior::{
    Assumption, Configuration, Connection, ModuleDefinition, Placement, PlatformService,
    Provenance, ReservedRegion, SafetyGate,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const IMPACT_ANALYZER_VERSION: &str = "marine-interior-impact-v1";
pub const EVIDENCE_POLICY: &str = "no_automatic_evidence_carry_forward";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Added,
    Removed,
    Modified,
    AmbiguousIdentity,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ArtifactChange {
    pub section: String,
    pub artifact_id: String,
    pub kind: ChangeKind,
    pub before_sha256: Option<String>,
    pub after_sha256: Option<String>,
    pub identity_ambiguous: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RevalidationDomain {
    pub domain: String,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ImpactReportBody {
    pub analyzer_version: String,
    pub from_configuration_id: String,
    pub to_configuration_id: String,
    pub from_configuration_sha256: String,
    pub to_configuration_sha256: String,
    pub changes: Vec<ArtifactChange>,
    pub revalidation_required: Vec<RevalidationDomain>,
    pub evidence_policy: String,
    pub disposition: String,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ImpactReport {
    pub body: ImpactReportBody,
    pub report_sha256: String,
}

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}

fn map_domain(
    impact: &mut BTreeMap<String, BTreeSet<String>>,
    domain: &str,
    reason: &str,
) {
    impact.entry(domain.to_owned()).or_default().insert(reason.to_owned());
}

fn apply_section_impacts(
    section: &str,
    artifact_id: &str,
    impact: &mut BTreeMap<String, BTreeSet<String>>,
) {
    let reason = format!("{section} artifact {artifact_id} changed");
    match section {
        "module_catalog" => {
            for domain in [
                "module_geometry",
                "interface_compatibility",
                "service_routing",
                "mass_and_centre_of_gravity",
                "structural_attachments",
                "fire_and_materials",
                "accessibility_and_egress",
                "ventilation_and_sanitation",
                "maintainability",
            ] {
                map_domain(impact, domain, &reason);
            }
        }
        "placements" => {
            for domain in [
                "layout_geometry",
                "service_clearances",
                "accessibility_and_egress",
                "service_routing",
                "mass_and_centre_of_gravity",
                "structural_attachments",
                "fire_and_watertight_boundaries",
                "emergency_equipment_access",
                "maintainability",
            ] {
                map_domain(impact, domain, &reason);
            }
        }
        "platform_services" => {
            for domain in [
                "interface_compatibility",
                "utility_capacity_and_protection",
                "service_routing",
                "isolation_and_maintainability",
            ] {
                map_domain(impact, domain, &reason);
            }
        }
        "connections" => {
            for domain in [
                "interface_compatibility",
                "utility_capacity_and_protection",
                "service_routing",
                "isolation_and_maintainability",
            ] {
                map_domain(impact, domain, &reason);
            }
        }
        "reserved_regions" => {
            for domain in [
                "layout_geometry",
                "accessibility_and_egress",
                "fire_and_watertight_boundaries",
                "emergency_equipment_access",
                "maintainability",
            ] {
                map_domain(impact, domain, &reason);
            }
        }
        "safety_gates" => {
            map_domain(impact, "safety_evidence_review", &reason);
        }
        "assumptions" => {
            for domain in [
                "assumption_validity",
                "evidence_applicability",
            ] {
                map_domain(impact, domain, &reason);
            }
        }
        "provenance" => {
            map_domain(impact, "evidence_provenance_review", &reason);
            map_domain(impact, "evidence_applicability", &reason);
        }
        "platform" | "coordinate_frame" => {
            for domain in [
                "layout_geometry",
                "service_clearances",
                "interface_compatibility",
                "service_routing",
                "mass_and_centre_of_gravity",
                "structural_attachments",
                "fire_and_watertight_boundaries",
                "accessibility_and_egress",
                "emergency_equipment_access",
                "maintainability",
                "safety_evidence_review",
            ] {
                map_domain(impact, domain, &reason);
            }
        }
        _ => {
            map_domain(impact, "configuration_review", &reason);
        }
    }
}

fn grouped<'a, T, F>(items: &'a [T], key: &F) -> BTreeMap<String, Vec<&'a T>>
where
    F: Fn(&T) -> String,
{
    let mut result = BTreeMap::new();
    for item in items {
        result.entry(key(item)).or_insert_with(Vec::new).push(item);
    }
    result
}

fn diff_collection<T, F>(
    section: &str,
    before: &[T],
    after: &[T],
    key: F,
    changes: &mut Vec<ArtifactChange>,
    impacts: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<(), serde_json::Error>
where
    T: Serialize,
    F: Fn(&T) -> String,
{
    let left = grouped(before, &key);
    let right = grouped(after, &key);
    let ids: BTreeSet<String> = left.keys().chain(right.keys()).cloned().collect();

    for artifact_id in ids {
        let left_items = left.get(&artifact_id);
        let right_items = right.get(&artifact_id);
        let duplicate = left_items.is_some_and(|items| items.len() != 1)
            || right_items.is_some_and(|items| items.len() != 1);

        let before_sha256 = match left_items {
            Some(items) => Some(digest(items)?),
            None => None,
        };
        let after_sha256 = match right_items {
            Some(items) => Some(digest(items)?),
            None => None,
        };

        let kind = if duplicate {
            ChangeKind::AmbiguousIdentity
        } else {
            match (before_sha256.as_deref(), after_sha256.as_deref()) {
                (None, Some(_)) => ChangeKind::Added,
                (Some(_), None) => ChangeKind::Removed,
                (Some(before), Some(after)) if before != after => ChangeKind::Modified,
                _ => continue,
            }
        };

        changes.push(ArtifactChange {
            section: section.to_owned(),
            artifact_id: artifact_id.clone(),
            kind,
            before_sha256,
            after_sha256,
            identity_ambiguous: duplicate,
        });
        apply_section_impacts(section, &artifact_id, impacts);

        if duplicate {
            for domain in [
                "artifact_identity_review",
                "evidence_applicability",
                "configuration_review",
            ] {
                map_domain(
                    impacts,
                    domain,
                    &format!("{section} identifier {artifact_id} is duplicated; change identity is ambiguous"),
                );
            }
        }
    }
    Ok(())
}

/// Compares two typed configurations and reports which analysis domains require revalidation.
///
/// The dependency map is deliberately conservative and versioned with this analyzer.
/// It is not a replacement for qualified engineering judgement or formal evidence review.
pub fn analyze_configuration_change(
    before: &Configuration,
    after: &Configuration,
) -> Result<ImpactReport, serde_json::Error> {
    let before_digest = digest(before)?;
    let after_digest = digest(after)?;
    let mut changes = Vec::new();
    let mut impacts: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    if digest(&before.platform)? != digest(&after.platform)? {
        changes.push(ArtifactChange {
            section: "platform".to_owned(),
            artifact_id: "platform".to_owned(),
            kind: ChangeKind::Modified,
            before_sha256: Some(digest(&before.platform)?),
            after_sha256: Some(digest(&after.platform)?),
            identity_ambiguous: false,
        });
        apply_section_impacts("platform", "platform", &mut impacts);
    }
    if digest(&before.coordinate_frame)? != digest(&after.coordinate_frame)? {
        changes.push(ArtifactChange {
            section: "coordinate_frame".to_owned(),
            artifact_id: "coordinate_frame".to_owned(),
            kind: ChangeKind::Modified,
            before_sha256: Some(digest(&before.coordinate_frame)?),
            after_sha256: Some(digest(&after.coordinate_frame)?),
            identity_ambiguous: false,
        });
        apply_section_impacts("coordinate_frame", "coordinate_frame", &mut impacts);
    }

    diff_collection::<ModuleDefinition, _>(
        "module_catalog", &before.module_catalog, &after.module_catalog,
        |value| value.module_id.clone(), &mut changes, &mut impacts,
    )?;
    diff_collection::<Placement, _>(
        "placements", &before.placements, &after.placements,
        |value| value.instance_id.clone(), &mut changes, &mut impacts,
    )?;
    diff_collection::<PlatformService, _>(
        "platform_services", &before.platform_services, &after.platform_services,
        |value| value.service_id.clone(), &mut changes, &mut impacts,
    )?;
    diff_collection::<Connection, _>(
        "connections", &before.connections, &after.connections,
        |value| value.connection_id.clone(), &mut changes, &mut impacts,
    )?;
    diff_collection::<ReservedRegion, _>(
        "reserved_regions", &before.reserved_regions, &after.reserved_regions,
        |value| value.region_id.clone(), &mut changes, &mut impacts,
    )?;
    diff_collection::<SafetyGate, _>(
        "safety_gates", &before.safety_gates, &after.safety_gates,
        |value| value.gate_id.clone(), &mut changes, &mut impacts,
    )?;
    diff_collection::<Assumption, _>(
        "assumptions", &before.assumptions, &after.assumptions,
        |value| value.assumption_id.clone(), &mut changes, &mut impacts,
    )?;
    diff_collection::<Provenance, _>(
        "provenance", &before.provenance, &after.provenance,
        |value| value.source_id.clone(), &mut changes, &mut impacts,
    )?;

    changes.sort_by(|a, b| {
        a.section
            .cmp(&b.section)
            .then(a.artifact_id.cmp(&b.artifact_id))
            .then(a.kind.cmp(&b.kind))
    });

    let revalidation_required: Vec<RevalidationDomain> = impacts
        .into_iter()
        .map(|(domain, reasons)| RevalidationDomain {
            domain,
            reasons: reasons.into_iter().collect(),
        })
        .collect();

    let disposition = if changes.is_empty() && before_digest == after_digest {
        "no_declared_change"
    } else if changes.iter().any(|change| change.identity_ambiguous) {
        "revalidation_required_identity_ambiguous"
    } else {
        "revalidation_required"
    };

    let body = ImpactReportBody {
        analyzer_version: IMPACT_ANALYZER_VERSION.to_owned(),
        from_configuration_id: before.configuration_id.clone(),
        to_configuration_id: after.configuration_id.clone(),
        from_configuration_sha256: before_digest,
        to_configuration_sha256: after_digest,
        changes,
        revalidation_required,
        evidence_policy: EVIDENCE_POLICY.to_owned(),
        disposition: disposition.to_owned(),
        claim_ceiling: "Change-impact analysis only. It does not validate the target configuration or approve re-use of prior engineering evidence.".to_owned(),
    };
    let report_sha256 = digest(&body)?;
    Ok(ImpactReport { body, report_sha256 })
}

/// Verifies that a report binds these exact configurations and has not been altered.
pub fn verify_impact_report(
    before: &Configuration,
    after: &Configuration,
    report: &ImpactReport,
) -> bool {
    let Ok(before_digest) = digest(before) else { return false; };
    let Ok(after_digest) = digest(after) else { return false; };
    let Ok(report_digest) = digest(&report.body) else { return false; };
    report.body.analyzer_version == IMPACT_ANALYZER_VERSION
        && report.body.from_configuration_sha256 == before_digest
        && report.body.to_configuration_sha256 == after_digest
        && report.body.from_configuration_id == before.configuration_id
        && report.body.to_configuration_id == after.configuration_id
        && report.body.evidence_policy == EVIDENCE_POLICY
        && report.report_sha256 == report_digest
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    const EXPEDITION: &str = include_str!("../fixtures/modular-interiors/expedition-research-layout-v1.json");

    fn live() -> Configuration { serde_json::from_str(LIVE).expect("live fixture parses") }
    fn expedition() -> Configuration { serde_json::from_str(EXPEDITION).expect("expedition fixture parses") }
    fn domains(report: &ImpactReport) -> Vec<&str> {
        report.body.revalidation_required.iter().map(|item| item.domain.as_str()).collect()
    }

    #[test]
    fn identical_configuration_is_a_deterministic_no_op() {
        let config = live();
        let one = analyze_configuration_change(&config, &config).expect("diff serializes");
        let two = analyze_configuration_change(&config, &config).expect("diff serializes");
        assert_eq!(one.body.disposition, "no_declared_change");
        assert!(one.body.changes.is_empty());
        assert!(one.body.revalidation_required.is_empty());
        assert_eq!(one.report_sha256, two.report_sha256);
        assert!(verify_impact_report(&config, &config, &one));
    }

    #[test]
    fn moved_module_invalidates_geometry_access_routes_and_mass_cg() {
        let before = live();
        let mut after = before.clone();
        after.placements.iter_mut().find(|p| p.instance_id == "CABIN-01")
            .expect("cabin exists").position_mm.x_mm += 150;
        let report = analyze_configuration_change(&before, &after).expect("diff serializes");
        assert_eq!(report.body.disposition, "revalidation_required");
        assert!(domains(&report).contains(&"layout_geometry"));
        assert!(domains(&report).contains(&"accessibility_and_egress"));
        assert!(domains(&report).contains(&"mass_and_centre_of_gravity"));
        assert!(report.body.changes.iter().any(|c| c.section == "placements" && c.artifact_id == "CABIN-01"));
        assert!(verify_impact_report(&before, &after, &report));
    }

    #[test]
    fn mass_property_change_requires_weight_and_related_review() {
        let before = live();
        let mut after = before.clone();
        after.module_catalog.iter_mut().find(|m| m.module_id == "SLEEP-POD")
            .expect("sleep pod exists").mass_properties.mass_kg = Some(311.0);
        let report = analyze_configuration_change(&before, &after).expect("diff serializes");
        assert!(domains(&report).contains(&"mass_and_centre_of_gravity"));
        assert!(domains(&report).contains(&"structural_attachments"));
        assert!(report.body.changes.iter().any(|c| c.section == "module_catalog"));
    }

    #[test]
    fn connection_and_service_edits_invalidate_interfaces_and_capacity() {
        let before = live();
        let mut after = before.clone();
        after.connections.pop();
        after.platform_services.iter_mut().find(|s| s.service_id == "AC-BUS")
            .expect("AC bus exists").capacity.value = Some(7000.0);
        let report = analyze_configuration_change(&before, &after).expect("diff serializes");
        assert!(domains(&report).contains(&"interface_compatibility"));
        assert!(domains(&report).contains(&"utility_capacity_and_protection"));
        assert!(report.body.changes.iter().any(|c| c.section == "connections"));
        assert!(report.body.changes.iter().any(|c| c.section == "platform_services"));
    }

    #[test]
    fn reserved_region_change_invalidates_egress_and_fire_boundary_review() {
        let before = live();
        let mut after = before.clone();
        after.reserved_regions.iter_mut().find(|r| r.region_id == "REGION-EGRESS-01")
            .expect("egress region exists").bounds_mm.y_max_mm = 900;
        let report = analyze_configuration_change(&before, &after).expect("diff serializes");
        assert!(domains(&report).contains(&"accessibility_and_egress"));
        assert!(domains(&report).contains(&"fire_and_watertight_boundaries"));
    }

    #[test]
    fn duplicate_identifiers_are_explicitly_ambiguous() {
        let before = live();
        let mut after = before.clone();
        after.placements.push(after.placements[0].clone());
        let report = analyze_configuration_change(&before, &after).expect("diff serializes");
        assert_eq!(report.body.disposition, "revalidation_required_identity_ambiguous");
        assert!(report.body.changes.iter().any(|c| c.identity_ambiguous));
        assert!(domains(&report).contains(&"artifact_identity_review"));
    }

    #[test]
    fn report_integrity_binds_source_and_target_revisions() {
        let before = live();
        let mut after = expedition();
        after.configuration_id = "LD-EXPEDITION-CHANGED-V2".to_owned();
        let report = analyze_configuration_change(&before, &after).expect("diff serializes");
        assert!(verify_impact_report(&before, &after, &report));
        let mut changed = after.clone();
        changed.configuration_name.push_str(" altered");
        assert!(!verify_impact_report(&before, &changed, &report));
        let mut tampered = report.clone();
        tampered.body.claim_ceiling.push_str(" altered");
        assert!(!verify_impact_report(&before, &after, &tampered));
    }

    #[test]
    fn artifact_order_does_not_change_change_report() {
        let before = live();
        let mut after = before.clone();
        after.module_catalog.reverse();
        after.placements.reverse();
        let report = analyze_configuration_change(&before, &after).expect("diff serializes");
        assert_eq!(report.body.disposition, "no_declared_change");
        assert!(report.body.changes.is_empty());
    }
}
