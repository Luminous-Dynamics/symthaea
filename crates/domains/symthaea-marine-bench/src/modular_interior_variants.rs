// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, conservative generation of single-module drop-in variants.
//!
//! Passing this preflight means only that the declared direct interface contract was
//! preserved. The candidate is still sent through the configuration validator and its
//! change-impact analyzer; neither result constitutes vessel approval.

use crate::modular_interior::{
    validate_configuration, Configuration, ModuleDefinition, ValidationReceipt,
};
use crate::modular_interior_impact::{
    analyze_configuration_change, ImpactReport,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const VARIANT_GENERATOR_VERSION: &str = "marine-interior-variants-v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleVariant {
    pub variant_id: String,
    pub replacement: ModuleDefinition,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreflightDisposition {
    Rejected,
    RequiresReconfiguration,
    DropInCompatible,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VariantDisposition {
    RejectedPreflight,
    RequiresReconfiguration,
    RejectedByValidator,
    ReviewRequired,
    CurrentChecksClear,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VariantFindingSeverity {
    Error,
    Review,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VariantFinding {
    pub code: String,
    pub severity: VariantFindingSeverity,
    pub subject: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GeneratedVariant {
    pub variant_id: String,
    pub candidate_id: String,
    pub disposition: VariantDisposition,
    pub baseline_configuration_sha256: String,
    pub candidate_configuration_sha256: Option<String>,
    pub preflight_findings: Vec<VariantFinding>,
    pub configuration: Option<Configuration>,
    pub validation_receipt: Option<ValidationReceipt>,
    pub impact_report: Option<ImpactReport>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VariantGenerationReportBody {
    pub generator_version: String,
    pub baseline_configuration_id: String,
    pub baseline_configuration_sha256: String,
    pub variant_input_manifest_sha256: String,
    pub evidence_policy: String,
    pub variants: Vec<GeneratedVariant>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VariantGenerationReport {
    pub body: VariantGenerationReportBody,
    pub report_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct VariantFingerprint {
    variant_id: String,
    replacement_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}

fn stable_id(value: &str) -> bool {
    (3..=64).contains(&value.len())
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn rating_rank(status: &str) -> Option<u8> {
    match status {
        "unknown" => Some(0),
        "design_assumption" => Some(1),
        "computed" => Some(2),
        "measured" | "source_verified" => Some(3),
        _ => None,
    }
}

fn module_evidence_rank(status: &str) -> Option<u8> {
    match status {
        "unknown" => Some(0),
        "design_assumption" => Some(1),
        "evidence_available" | "not_applicable" => Some(2),
        "authority_reviewed" => Some(3),
        _ => None,
    }
}

fn add_finding(
    findings: &mut Vec<VariantFinding>,
    code: &str,
    severity: VariantFindingSeverity,
    subject: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(VariantFinding {
        code: code.to_owned(),
        severity,
        subject: subject.into(),
        message: message.into(),
    });
}

fn preflight(
    baseline: &Configuration,
    variant: &ModuleVariant,
    duplicate_variant_id: bool,
) -> (PreflightDisposition, Vec<VariantFinding>) {
    let mut findings = Vec::new();
    let replacement = &variant.replacement;

    if duplicate_variant_id {
        add_finding(
            &mut findings,
            "DUPLICATE_VARIANT_ID",
            VariantFindingSeverity::Error,
            &variant.variant_id,
            "variant IDs must be unique within a generation batch",
        );
    }
    if !stable_id(&variant.variant_id) {
        add_finding(
            &mut findings,
            "INVALID_VARIANT_ID",
            VariantFindingSeverity::Error,
            &variant.variant_id,
            "variant ID must be 3-64 ASCII letters, digits, dots, underscores or hyphens",
        );
    }

    let matching_modules: Vec<&ModuleDefinition> = baseline.module_catalog.iter()
        .filter(|module| module.module_id == replacement.module_id)
        .collect();
    let original = match matching_modules.as_slice() {
        [module] => Some(*module),
        [] => {
            add_finding(
                &mut findings,
                "TARGET_MODULE_NOT_FOUND",
                VariantFindingSeverity::Error,
                &replacement.module_id,
                "replacement module ID does not exist in the baseline catalog",
            );
            None
        }
        _ => {
            add_finding(
                &mut findings,
                "TARGET_MODULE_ID_AMBIGUOUS",
                VariantFindingSeverity::Error,
                &replacement.module_id,
                "baseline catalog contains multiple definitions with this module ID",
            );
            None
        }
    };

    if replacement.revision == 0
        || replacement.envelope_mm.length_mm <= 0
        || replacement.envelope_mm.width_mm <= 0
        || replacement.envelope_mm.height_mm <= 0
    {
        add_finding(
            &mut findings,
            "INVALID_REPLACEMENT_GEOMETRY_OR_REVISION",
            VariantFindingSeverity::Error,
            &replacement.module_id,
            "replacement revision and envelope dimensions must be positive",
        );
    }
    if replacement.ports.is_empty() {
        add_finding(
            &mut findings,
            "EMPTY_REPLACEMENT_PORT_CATALOG",
            VariantFindingSeverity::Error,
            &replacement.module_id,
            "replacement must declare its typed interface ports",
        );
    }
    let mut port_ids = BTreeSet::new();
    for port in &replacement.ports {
        if !stable_id(&port.port_id) || !port_ids.insert(port.port_id.clone()) {
            add_finding(
                &mut findings,
                "INVALID_OR_DUPLICATE_REPLACEMENT_PORT_ID",
                VariantFindingSeverity::Error,
                format!("{}.{}", replacement.module_id, port.port_id),
                "replacement port IDs must be stable and unique",
            );
        }
    }
    let c = &replacement.service_clearance_mm;
    if [c.forward, c.aft, c.port, c.starboard, c.above].iter().any(|value| *value < 0) {
        add_finding(
            &mut findings,
            "NEGATIVE_REPLACEMENT_SERVICE_CLEARANCE",
            VariantFindingSeverity::Error,
            &replacement.module_id,
            "replacement service-clearance margins cannot be negative",
        );
    }

    if let Some(original) = original {
        if replacement.revision <= original.revision {
            add_finding(
                &mut findings,
                "NON_MONOTONIC_MODULE_REVISION",
                VariantFindingSeverity::Error,
                &replacement.module_id,
                "replacement revision must be greater than the baseline revision",
            );
        }

        if replacement.ports.len() != original.ports.len() {
            add_finding(
                &mut findings,
                "PORT_SET_CHANGED",
                VariantFindingSeverity::Review,
                &replacement.module_id,
                "direct drop-in variants must preserve the complete port set; adding/removing ports requires explicit reconfiguration",
            );
        }
        for old_port in &original.ports {
            let matches: Vec<_> = replacement.ports.iter()
                .filter(|port| port.port_id == old_port.port_id)
                .collect();
            let new_port = match matches.as_slice() {
                [port] => Some(*port),
                [] => {
                    add_finding(
                        &mut findings,
                        "CONNECTED_PORT_REMOVED",
                        VariantFindingSeverity::Review,
                        format!("{}.{}", replacement.module_id, old_port.port_id),
                        "replacement removes a baseline port; adapter or connection redesign is required",
                    );
                    None
                }
                _ => {
                    add_finding(
                        &mut findings,
                        "REPLACEMENT_PORT_AMBIGUOUS",
                        VariantFindingSeverity::Error,
                        format!("{}.{}", replacement.module_id, old_port.port_id),
                        "replacement contains duplicate definitions for a port ID",
                    );
                    None
                }
            };
            let Some(new_port) = new_port else { continue; };

            if old_port.medium != new_port.medium
                || old_port.direction != new_port.direction
                || old_port.connector_profile != new_port.connector_profile
                || old_port.criticality != new_port.criticality
                || old_port.rating.unit != new_port.rating.unit
            {
                add_finding(
                    &mut findings,
                    "PORT_CONTRACT_CHANGED",
                    VariantFindingSeverity::Review,
                    format!("{}.{}", replacement.module_id, old_port.port_id),
                    "medium, direction, connector profile, criticality and rating unit must remain compatible for an automatic drop-in",
                );
            }
            match (old_port.rating.value, new_port.rating.value) {
                (Some(old_value), Some(new_value))
                    if new_value.is_finite() && new_value >= old_value => {}
                (Some(_), Some(new_value)) if !new_value.is_finite() || new_value < 0.0 => {
                    add_finding(
                        &mut findings,
                        "INVALID_REPLACEMENT_PORT_RATING",
                        VariantFindingSeverity::Error,
                        format!("{}.{}", replacement.module_id, old_port.port_id),
                        "replacement rating must be finite and non-negative",
                    );
                }
                (Some(_), Some(_)) => {
                    add_finding(
                        &mut findings,
                        "PORT_RATING_DOWNGRADE",
                        VariantFindingSeverity::Review,
                        format!("{}.{}", replacement.module_id, old_port.port_id),
                        "replacement port rating is lower than the baseline rating",
                    );
                }
                _ => {
                    add_finding(
                        &mut findings,
                        "PORT_RATING_UNKNOWN",
                        VariantFindingSeverity::Review,
                        format!("{}.{}", replacement.module_id, old_port.port_id),
                        "baseline/replacement port ratings are incomplete; drop-in compatibility is indeterminate",
                    );
                }
            }
            match (rating_rank(&old_port.rating.status), rating_rank(&new_port.rating.status)) {
                (Some(old_rank), Some(new_rank)) if new_rank >= old_rank => {}
                _ => add_finding(
                    &mut findings,
                    "PORT_EVIDENCE_DOWNGRADE",
                    VariantFindingSeverity::Review,
                    format!("{}.{}", replacement.module_id, old_port.port_id),
                    "replacement port rating evidence status is invalid or weaker than baseline",
                ),
            }
        }

        for (label, old, new) in [
            ("fire/material", &original.fire_material_status, &replacement.fire_material_status),
            ("mounting", &original.mounting_status, &replacement.mounting_status),
            ("electrical", &original.electrical_status, &replacement.electrical_status),
            ("sanitary", &original.sanitary_status, &replacement.sanitary_status),
        ] {
            match (module_evidence_rank(old), module_evidence_rank(new)) {
                (Some(old_rank), Some(new_rank)) if new_rank >= old_rank => {}
                _ => add_finding(
                    &mut findings,
                    "MODULE_EVIDENCE_DOWNGRADE",
                    VariantFindingSeverity::Review,
                    format!("{} {label}", replacement.module_id),
                    "replacement safety/system evidence status is invalid or weaker than baseline",
                ),
            }
        }

        if let (Some(old_mass), Some(new_mass)) =
            (original.mass_properties.mass_kg, replacement.mass_properties.mass_kg)
        {
            if !new_mass.is_finite() || new_mass < 0.0 {
                add_finding(
                    &mut findings,
                    "INVALID_REPLACEMENT_MASS",
                    VariantFindingSeverity::Error,
                    &replacement.module_id,
                    "replacement mass must be finite and non-negative",
                );
            }
            if old_mass.is_finite() && new_mass.is_finite() && new_mass != old_mass {
                // A changed mass is permitted, but downstream weight/CG and structural
                // revalidation is always emitted by the configuration impact analyzer.
            }
        } else {
            add_finding(
                &mut findings,
                "MASS_PROPERTIES_INCOMPLETE",
                VariantFindingSeverity::Review,
                &replacement.module_id,
                "baseline or replacement mass is unknown; direct catalog replacement requires engineering review",
            );
        }
        if let (Some(old_rank), Some(new_rank)) = (
            rating_rank(&original.mass_properties.status),
            rating_rank(&replacement.mass_properties.status),
        ) {
            if new_rank < old_rank {
                add_finding(
                    &mut findings,
                    "MASS_EVIDENCE_DOWNGRADE",
                    VariantFindingSeverity::Review,
                    &replacement.module_id,
                    "replacement mass evidence status is weaker than baseline",
                );
            }
        } else {
            add_finding(
                &mut findings,
                "MASS_EVIDENCE_STATUS_INVALID",
                VariantFindingSeverity::Review,
                &replacement.module_id,
                "baseline or replacement mass evidence status is invalid",
            );
        }
        let old_cg = &original.mass_properties.local_cg_mm;
        let new_cg = &replacement.mass_properties.local_cg_mm;
        if (old_cg.x_mm.is_some() && new_cg.x_mm.is_none())
            || (old_cg.y_mm.is_some() && new_cg.y_mm.is_none())
            || (old_cg.z_mm.is_some() && new_cg.z_mm.is_none())
        {
            add_finding(
                &mut findings,
                "CENTRE_OF_GRAVITY_DATA_DOWNGRADE",
                VariantFindingSeverity::Review,
                &replacement.module_id,
                "replacement omits centre-of-gravity coordinates known for the baseline",
            );
        }
    }

    if findings.iter().any(|finding| finding.severity == VariantFindingSeverity::Error) {
        (PreflightDisposition::Rejected, findings)
    } else if findings.iter().any(|finding| finding.severity == VariantFindingSeverity::Review) {
        (PreflightDisposition::RequiresReconfiguration, findings)
    } else {
        (PreflightDisposition::DropInCompatible, findings)
    }
}

fn input_fingerprints(
    variants: &[ModuleVariant],
) -> Result<Vec<VariantFingerprint>, serde_json::Error> {
    let mut fingerprints = Vec::with_capacity(variants.len());
    for variant in variants {
        fingerprints.push(VariantFingerprint {
            variant_id: variant.variant_id.clone(),
            replacement_sha256: digest(&variant.replacement)?,
        });
    }
    fingerprints.sort_by(|a, b| a.variant_id.cmp(&b.variant_id)
        .then(a.replacement_sha256.cmp(&b.replacement_sha256)));
    Ok(fingerprints)
}

/// Generates single-module replacement candidates whose declared interface contract
/// is unchanged. Compatible candidates are immediately checked by the configuration
/// validator and revalidation-impact analyzer. A preflight pass is not a safety approval.
pub fn generate_drop_in_variants(
    baseline: &Configuration,
    variants: &[ModuleVariant],
) -> Result<VariantGenerationReport, serde_json::Error> {
    let baseline_sha256 = digest(baseline)?;
    let fingerprints = input_fingerprints(variants)?;
    let variant_input_manifest_sha256 = digest(&fingerprints)?;
    let mut variant_id_counts = BTreeMap::<String, usize>::new();
    for variant in variants {
        *variant_id_counts.entry(variant.variant_id.clone()).or_default() += 1;
    }

    let mut sorted_variants = Vec::with_capacity(variants.len());
    for variant in variants {
        sorted_variants.push((variant.clone(), digest(&variant.replacement)?));
    }
    sorted_variants.sort_by(|(a, a_digest), (b, b_digest)| {
        a.variant_id.cmp(&b.variant_id).then(a_digest.cmp(b_digest))
    });

    let mut output = Vec::with_capacity(sorted_variants.len());
    for (variant, replacement_sha256) in &sorted_variants {
        let duplicate_id = variant_id_counts.get(&variant.variant_id).copied().unwrap_or(0) > 1;
        let (preflight_disposition, preflight_findings) =
            preflight(baseline, variant, duplicate_id);
        let seed = format!(
            "{}:{}:{}:{}",
            baseline_sha256,
            variant.variant_id,
            variant.replacement.revision,
            replacement_sha256
        );
        let candidate_suffix = digest(&seed)?[..16].to_owned();
        let candidate_id = format!("modvar-{candidate_suffix}");

        if preflight_disposition != PreflightDisposition::DropInCompatible {
            output.push(GeneratedVariant {
                variant_id: variant.variant_id.clone(),
                candidate_id,
                disposition: if preflight_disposition == PreflightDisposition::Rejected {
                    VariantDisposition::RejectedPreflight
                } else {
                    VariantDisposition::RequiresReconfiguration
                },
                baseline_configuration_sha256: baseline_sha256.clone(),
                candidate_configuration_sha256: None,
                preflight_findings,
                configuration: None,
                validation_receipt: None,
                impact_report: None,
            });
            continue;
        }

        let mut candidate = baseline.clone();
        if let Some(module) = candidate.module_catalog.iter_mut()
            .find(|module| module.module_id == variant.replacement.module_id)
        {
            *module = variant.replacement.clone();
        } else {
            // Normally unreachable because preflight established a unique target.
            output.push(GeneratedVariant {
                variant_id: variant.variant_id.clone(),
                candidate_id,
                disposition: VariantDisposition::RejectedPreflight,
                baseline_configuration_sha256: baseline_sha256.clone(),
                candidate_configuration_sha256: None,
                preflight_findings: vec![VariantFinding {
                    code: "TARGET_DISAPPEARED".to_owned(),
                    severity: VariantFindingSeverity::Error,
                    subject: variant.replacement.module_id.clone(),
                    message: "target module disappeared between preflight and materialization".to_owned(),
                }],
                configuration: None,
                validation_receipt: None,
                impact_report: None,
            });
            continue;
        }
        for placement in &mut candidate.placements {
            if placement.module_id == variant.replacement.module_id {
                placement.module_revision = variant.replacement.revision;
            }
        }
        candidate.configuration_id = format!("LD-MODVAR-{candidate_suffix}");
        candidate.configuration_name = format!(
            "{} / {}",
            baseline.configuration_name,
            variant.variant_id
        );
        candidate.customization_intent = format!(
            "{} Candidate module variant {}. Generated for screening; downstream validation remains mandatory.",
            baseline.customization_intent,
            variant.variant_id
        );

        let candidate_sha256 = digest(&candidate)?;
        let validation_receipt = validate_configuration(&candidate)?;
        let impact_report = analyze_configuration_change(baseline, &candidate)?;
        let disposition = match validation_receipt.body.overall_disposition.as_str() {
            "rejected" => VariantDisposition::RejectedByValidator,
            "indeterminate" => VariantDisposition::ReviewRequired,
            "candidate_only" => VariantDisposition::CurrentChecksClear,
            _ => VariantDisposition::ReviewRequired,
        };

        output.push(GeneratedVariant {
            variant_id: variant.variant_id.clone(),
            candidate_id,
            disposition,
            baseline_configuration_sha256: baseline_sha256.clone(),
            candidate_configuration_sha256: Some(candidate_sha256),
            preflight_findings,
            configuration: Some(candidate),
            validation_receipt: Some(validation_receipt),
            impact_report: Some(impact_report),
        });
    }

    output.sort_by(|a, b| a.variant_id.cmp(&b.variant_id)
        .then(a.candidate_id.cmp(&b.candidate_id)));
    let body = VariantGenerationReportBody {
        generator_version: VARIANT_GENERATOR_VERSION.to_owned(),
        baseline_configuration_id: baseline.configuration_id.clone(),
        baseline_configuration_sha256: baseline_sha256,
        variant_input_manifest_sha256,
        evidence_policy: "drop_in_contract_preservation_no_automatic_evidence_carry_forward".to_owned(),
        variants: output,
        claim_ceiling: "Variant generation and declared-interface preflight only. Generated candidates still require vessel-specific geometry, mass/CG, structural, systems, access, safety and regulatory validation.".to_owned(),
    };
    let report_sha256 = digest(&body)?;
    Ok(VariantGenerationReport { body, report_sha256 })
}

/// Recomputes generation, validation and impact analysis from the exact inputs.
pub fn verify_variant_generation_report(
    baseline: &Configuration,
    variants: &[ModuleVariant],
    report: &VariantGenerationReport,
) -> bool {
    let Ok(recomputed) = generate_drop_in_variants(baseline, variants) else {
        return false;
    };
    recomputed.report_sha256 == report.report_sha256
        && digest(&recomputed.body).ok() == digest(&report.body).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");

    fn live() -> Configuration {
        serde_json::from_str(LIVE).expect("live fixture parses")
    }

    fn sleep_pod(configuration: &Configuration) -> ModuleDefinition {
        configuration.module_catalog.iter().find(|module| module.module_id == "SLEEP-POD")
            .expect("fixture has sleep pod").clone()
    }

    fn direct_drop_in(configuration: &Configuration, variant_id: &str, revision: u32) -> ModuleVariant {
        let mut replacement = sleep_pod(configuration);
        replacement.revision = revision;
        replacement.name = format!("{} alternative", replacement.name);
        replacement.envelope_mm.length_mm += 100;
        ModuleVariant { variant_id: variant_id.to_owned(), replacement }
    }

    #[test]
    fn compatible_variant_is_generated_but_stays_evidence_indeterminate() {
        let baseline = live();
        let variants = vec![direct_drop_in(&baseline, "SLEEP-POD-V2", 2)];
        let report = generate_drop_in_variants(&baseline, &variants).expect("report serializes");
        let generated = &report.body.variants[0];
        assert_eq!(generated.disposition, VariantDisposition::ReviewRequired);
        assert!(generated.configuration.is_some());
        assert!(generated.validation_receipt.as_ref().is_some_and(|receipt|
            receipt.body.overall_disposition == "indeterminate"));
        assert!(generated.impact_report.is_some());
        assert!(verify_variant_generation_report(&baseline, &variants, &report));
    }

    #[test]
    fn module_revision_must_increase() {
        let baseline = live();
        let variants = vec![direct_drop_in(&baseline, "SLEEP-POD-V1", 1)];
        let report = generate_drop_in_variants(&baseline, &variants).expect("report serializes");
        assert_eq!(report.body.variants[0].disposition, VariantDisposition::RejectedPreflight);
        assert!(report.body.variants[0].preflight_findings.iter()
            .any(|finding| finding.code == "NON_MONOTONIC_MODULE_REVISION"));
    }

    #[test]
    fn renamed_or_removed_port_requires_explicit_reconfiguration() {
        let baseline = live();
        let mut variant = direct_drop_in(&baseline, "SLEEP-POD-PORTS", 2);
        variant.replacement.ports.retain(|port| port.port_id != "AC_IN");
        let report = generate_drop_in_variants(&baseline, &[variant]).expect("report serializes");
        assert_eq!(report.body.variants[0].disposition, VariantDisposition::RequiresReconfiguration);
        assert!(report.body.variants[0].configuration.is_none());
        assert!(report.body.variants[0].preflight_findings.iter()
            .any(|finding| finding.code == "CONNECTED_PORT_REMOVED"));
    }

    #[test]
    fn rating_downgrade_requires_reconfiguration() {
        let baseline = live();
        let mut variant = direct_drop_in(&baseline, "SLEEP-POD-LOWER-RATING", 2);
        variant.replacement.ports.iter_mut().find(|port| port.port_id == "AC_IN")
            .expect("AC port exists").rating.value = Some(100.0);
        let report = generate_drop_in_variants(&baseline, &[variant]).expect("report serializes");
        assert_eq!(report.body.variants[0].disposition, VariantDisposition::RequiresReconfiguration);
        assert!(report.body.variants[0].preflight_findings.iter()
            .any(|finding| finding.code == "PORT_RATING_DOWNGRADE"));
    }

    #[test]
    fn duplicate_variant_ids_are_rejected_without_silent_collapse() {
        let baseline = live();
        let one = direct_drop_in(&baseline, "SLEEP-POD-DUP", 2);
        let mut two = direct_drop_in(&baseline, "SLEEP-POD-DUP", 3);
        two.replacement.envelope_mm.width_mm += 50;
        let report = generate_drop_in_variants(&baseline, &[one, two]).expect("report serializes");
        assert_eq!(report.body.variants.len(), 2);
        assert!(report.body.variants.iter().all(|variant|
            variant.disposition == VariantDisposition::RejectedPreflight));
    }

    #[test]
    fn generation_report_is_order_independent_and_detects_tampering() {
        let baseline = live();
        let a = direct_drop_in(&baseline, "SLEEP-POD-ALT-A", 2);
        let b = direct_drop_in(&baseline, "SLEEP-POD-ALT-B", 3);
        let first = vec![a.clone(), b.clone()];
        let second = vec![b, a];
        let report_a = generate_drop_in_variants(&baseline, &first).expect("report serializes");
        let report_b = generate_drop_in_variants(&baseline, &second).expect("report serializes");
        assert_eq!(report_a.report_sha256, report_b.report_sha256);
        assert!(verify_variant_generation_report(&baseline, &second, &report_a));
        let mut tampered = report_a;
        tampered.body.claim_ceiling.push_str(" altered");
        assert!(!verify_variant_generation_report(&baseline, &second, &tampered));
    }
}
