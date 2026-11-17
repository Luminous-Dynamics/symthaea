// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed, per-zone requirement admission for Atelier robotic Lightshell.
//!
//! This module records *what a zone must do* and what evidence will be needed.
//! It does not choose materials, predict textile performance, qualify a cover,
//! or authorize robot motion.

use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneSpecManifest {
    pub schema_version: String,
    pub spec_id: String,
    pub state: String,
    pub platform: PlatformIdentity,
    pub operating_domain: Vec<String>,
    pub zones: Vec<ZoneBrief>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformIdentity {
    pub platform_id: String,
    pub article_revision: Option<String>,
    pub geometry_digest: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneBrief {
    pub zone_id: String,
    pub robot_zone_ref: Option<String>,
    pub frame_ref: Option<String>,
    pub functions: Vec<String>,
    pub requirements: Vec<ZoneRequirement>,
    pub keep_out_refs: Vec<String>,
    pub clearance_review_ref: Option<String>,
    pub service: ServicePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneRequirement {
    pub function_id: String,
    pub metric_id: String,
    pub unit: String,
    pub direction: String,
    pub hard_gate: bool,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub method_ref: Option<String>,
    pub acceptance_rule_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServicePolicy {
    pub replaceable: bool,
    pub interface_ref: Option<String>,
    pub inspection_access: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub unknowns_fail_closed: bool,
    pub visual_estimates_are_non_qualifying: bool,
    pub simulation_is_not_physical_qualification: bool,
    pub supplier_claims_are_not_independent_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub zone_count: usize,
    pub requirement_count: usize,
    pub requirements_frozen: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// A frozen contract is only a complete, reviewable requirement brief.
    /// It is not proof of material performance, robot safety or certification.
    pub fn is_freezable(&self) -> bool {
        self.is_valid() && self.requirements_frozen
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: ZoneSpecManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl ZoneSpecManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut zone_ids = HashSet::new();
        let mut requirement_count = 0usize;

        if self.schema_version != "0.1.0" {
            push(
                &mut findings,
                "unsupported_schema_version",
                "schema_version",
                "Only zone-spec schema version 0.1.0 is admitted.",
            );
        }
        if is_blank(&self.spec_id) || is_blank(&self.platform.platform_id) {
            push(
                &mut findings,
                "missing_spec_identity",
                "spec_id/platform.platform_id",
                "Specification and platform identities must be non-empty.",
            );
        }
        if !matches!(
            self.state.as_str(),
            "proposed" | "requirements_pending" | "frozen"
        ) {
            push(
                &mut findings,
                "invalid_spec_state",
                "state",
                "State must be proposed, requirements_pending or frozen.",
            );
        }
        if let Some(digest) = self.platform.geometry_digest.as_deref() {
            if !is_sha256_digest(digest) {
                push(
                    &mut findings,
                    "invalid_geometry_digest",
                    "platform.geometry_digest",
                    "A supplied geometry digest must use sha256: followed by 64 hexadecimal characters.",
                );
            }
        }

        for (index, item) in self.operating_domain.iter().enumerate() {
            if is_blank(item) {
                push(
                    &mut findings,
                    "blank_operating_domain",
                    format!("operating_domain[{index}]"),
                    "Operating-domain entries must be non-empty.",
                );
            }
        }

        if self.zones.is_empty() {
            push(
                &mut findings,
                "no_zones",
                "zones",
                "At least one robot zone must be declared.",
            );
        }

        for (index, zone) in self.zones.iter().enumerate() {
            let path = format!("zones[{index}]");
            if is_blank(&zone.zone_id) {
                push(
                    &mut findings,
                    "blank_zone_id",
                    format!("{path}.zone_id"),
                    "Zone identity must be non-empty.",
                );
            }
            if !zone_ids.insert(zone.zone_id.as_str()) {
                push(
                    &mut findings,
                    "duplicate_zone_id",
                    format!("{path}.zone_id"),
                    "Zone IDs must be unique within a specification.",
                );
            }

            let mut functions = HashSet::new();
            for (function_index, function) in zone.functions.iter().enumerate() {
                if !is_known_function(function) {
                    push(
                        &mut findings,
                        "unknown_function",
                        format!("{path}.functions[{function_index}]"),
                        "Unknown zone function.",
                    );
                }
                if !functions.insert(function.as_str()) {
                    push(
                        &mut findings,
                        "duplicate_function",
                        format!("{path}.functions[{function_index}]"),
                        "A zone function may be declared only once.",
                    );
                }
            }
            if zone.functions.is_empty() {
                push(
                    &mut findings,
                    "zone_without_function",
                    format!("{path}.functions"),
                    "Each zone must declare at least one functional role.",
                );
            }

            let mut requirement_keys = HashSet::new();
            let mut measured_functions = HashSet::new();
            for (requirement_index, requirement) in zone.requirements.iter().enumerate() {
                requirement_count += 1;
                let requirement_path = format!("{path}.requirements[{requirement_index}]");

                if !is_known_function(&requirement.function_id) {
                    push(
                        &mut findings,
                        "unknown_requirement_function",
                        format!("{requirement_path}.function_id"),
                        "Requirement references an unknown functional role.",
                    );
                }
                if !is_known_metric(&requirement.metric_id) {
                    push(
                        &mut findings,
                        "unknown_metric",
                        format!("{requirement_path}.metric_id"),
                        "Metric is not in the Lightshell zone metric vocabulary.",
                    );
                } else {
                    if !unit_matches_metric(&requirement.metric_id, &requirement.unit) {
                        push(
                            &mut findings,
                            "metric_unit_mismatch",
                            format!("{requirement_path}.unit"),
                            "Metric unit does not match the canonical unit for this metric.",
                        );
                    }
                    if direction_for_metric(&requirement.metric_id)
                        != Some(requirement.direction.as_str())
                    {
                        push(
                            &mut findings,
                            "metric_direction_mismatch",
                            format!("{requirement_path}.direction"),
                            "Comparison direction does not match the declared metric semantics.",
                        );
                    }
                    if function_supports_metric(
                        &requirement.function_id,
                        &requirement.metric_id,
                    ) {
                        measured_functions.insert(requirement.function_id.as_str());
                    } else {
                        push(
                            &mut findings,
                            "function_metric_mismatch",
                            &requirement_path,
                            "The metric does not provide evidence for the requirement's functional role.",
                        );
                    }
                }

                if !requirement_keys.insert((
                    requirement.function_id.as_str(),
                    requirement.metric_id.as_str(),
                )) {
                    push(
                        &mut findings,
                        "duplicate_zone_requirement",
                        &requirement_path,
                        "A function/metric pair may appear only once per zone.",
                    );
                }

                if is_blank(&requirement.unit) {
                    push(
                        &mut findings,
                        "blank_metric_unit",
                        format!("{requirement_path}.unit"),
                        "A metric unit must be explicit.",
                    );
                }
                if !bounds_are_consistent(requirement, self.state == "frozen") {
                    push(
                        &mut findings,
                        "invalid_metric_bounds",
                        format!("{requirement_path}.lower_bound/upper_bound"),
                        "Bounds must be ordered and must match the comparison direction.",
                    );
                }

                if self.state == "frozen" {
                    if is_blank_option(&requirement.method_ref) {
                        push(
                            &mut findings,
                            "frozen_requirement_missing_method",
                            format!("{requirement_path}.method_ref"),
                            "A frozen requirement must cite a measurement method or protocol.",
                        );
                    }
                    if is_blank_option(&requirement.acceptance_rule_ref) {
                        push(
                            &mut findings,
                            "frozen_requirement_missing_acceptance",
                            format!("{requirement_path}.acceptance_rule_ref"),
                            "A frozen requirement must cite an approved acceptance rule.",
                        );
                    }
                    if requirement.hard_gate
                        && (requirement.lower_bound.is_none()
                            && requirement.upper_bound.is_none())
                    {
                        push(
                            &mut findings,
                            "hard_gate_without_limit",
                            &requirement_path,
                            "A frozen hard gate requires at least one explicit numeric bound.",
                        );
                    }
                }
            }

            if zone.requirements.is_empty() {
                push(
                    &mut findings,
                    "zone_without_requirement",
                    format!("{path}.requirements"),
                    "Each zone function must have at least one linked measurement requirement.",
                );
            }
            for function in &zone.functions {
                if is_known_function(function) && !measured_functions.contains(function.as_str()) {
                    push(
                        &mut findings,
                        "function_without_measurement",
                        format!("{path}.functions"),
                        format!("Functional role {function} has no compatible measurement requirement."),
                    );
                }
            }

            if self.state == "frozen" {
                if is_blank_option(&zone.robot_zone_ref) {
                    push(
                        &mut findings,
                        "frozen_zone_missing_cad_ref",
                        format!("{path}.robot_zone_ref"),
                        "A frozen zone must bind to a robot CAD zone or surface identity.",
                    );
                }
                if is_blank_option(&zone.frame_ref) {
                    push(
                        &mut findings,
                        "frozen_zone_missing_frame",
                        format!("{path}.frame_ref"),
                        "A frozen zone must bind to an explicit coordinate frame.",
                    );
                }
                if is_blank_option(&zone.clearance_review_ref) {
                    push(
                        &mut findings,
                        "frozen_zone_missing_clearance_review",
                        format!("{path}.clearance_review_ref"),
                        "A frozen zone must cite its keep-out and clearance review.",
                    );
                }
                if zone.service.replaceable
                    && (is_blank_option(&zone.service.interface_ref)
                        || !zone.service.inspection_access)
                {
                    push(
                        &mut findings,
                        "replaceable_zone_not_serviceable",
                        format!("{path}.service"),
                        "A replaceable zone requires a named attachment interface and inspection access.",
                    );
                }
            }
        }

        if self.state == "frozen" {
            if is_blank_option(&self.platform.article_revision) {
                push(
                    &mut findings,
                    "frozen_platform_missing_revision",
                    "platform.article_revision",
                    "A frozen specification must bind an exact robot article revision.",
                );
            }
            if self.platform.geometry_digest.is_none() {
                push(
                    &mut findings,
                    "frozen_platform_missing_geometry_digest",
                    "platform.geometry_digest",
                    "A frozen specification must bind the source geometry digest.",
                );
            }
            if self.operating_domain.is_empty() {
                push(
                    &mut findings,
                    "frozen_spec_missing_operating_domain",
                    "operating_domain",
                    "A frozen specification must declare its operating domain.",
                );
            }
        }

        if !self.evidence_policy.unknowns_fail_closed
            || !self.evidence_policy.visual_estimates_are_non_qualifying
            || !self.evidence_policy.simulation_is_not_physical_qualification
            || !self.evidence_policy.supplier_claims_are_not_independent_qualification
        {
            push(
                &mut findings,
                "unsafe_evidence_policy",
                "evidence_policy",
                "Unknowns must fail closed; visual estimates, simulation and supplier claims cannot independently qualify a physical article.",
            );
        }

        ValidationReport {
            requirements_frozen: self.state == "frozen" && findings.is_empty(),
            findings,
            zone_count: self.zones.len(),
            requirement_count,
        }
    }
}

fn push(
    findings: &mut Vec<Finding>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(Finding {
        code,
        path: path.into(),
        message: message.into(),
    });
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_blank_option(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_blank)
}

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_known_function(value: &str) -> bool {
    matches!(
        value,
        "impact_compression"
            | "vibration_isolation"
            | "airflow"
            | "thermal_management"
            | "articulation"
            | "abrasion"
            | "fluid_chemical_barrier"
            | "cleanability"
            | "sensor_accommodation"
            | "repairability"
    )
}

fn is_known_metric(value: &str) -> bool {
    metric_definition(value).is_some()
}

fn metric_definition(value: &str) -> Option<(&'static str, &'static str)> {
    match value {
        "compression_energy_j_g" => Some(("J/g", "higher_is_better")),
        "transmitted_force_n" => Some(("N", "lower_is_better")),
        "vibration_transmissibility_ratio" => Some(("ratio", "lower_is_better")),
        "air_permeability_mm_s" => Some(("mm/s", "higher_is_better")),
        "temperature_delta_c" => Some(("degC", "within_declared_band")),
        "joint_rom_delta_deg" => Some(("deg", "within_declared_band")),
        "flex_fatigue_cycles" => Some(("cycles", "higher_is_better")),
        "minimum_clearance_mm" => Some(("mm", "higher_is_better")),
        "abrasion_cycles" => Some(("cycles", "higher_is_better")),
        "fluid_ingress_mass_g_m2" => Some(("g/m2", "lower_is_better")),
        "chemical_mass_change_pct" => Some(("%", "within_declared_band")),
        "cleaning_residue_fraction" => Some(("fraction", "lower_is_better")),
        "sensor_occlusion_fraction" => Some(("fraction", "lower_is_better")),
        "repair_time_min" => Some(("min", "lower_is_better")),
        "added_mass_g" => Some(("g", "lower_is_better")),
        "seam_strength_n" => Some(("N", "higher_is_better")),
        _ => None,
    }
}

fn unit_matches_metric(metric_id: &str, unit: &str) -> bool {
    metric_definition(metric_id).is_some_and(|(canonical, _)| canonical == unit)
}

fn direction_for_metric(metric_id: &str) -> Option<&'static str> {
    metric_definition(metric_id).map(|(_, direction)| direction)
}

fn function_supports_metric(function_id: &str, metric_id: &str) -> bool {
    match function_id {
        "impact_compression" => {
            matches!(metric_id, "compression_energy_j_g" | "transmitted_force_n")
        }
        "vibration_isolation" => metric_id == "vibration_transmissibility_ratio",
        "airflow" => metric_id == "air_permeability_mm_s",
        "thermal_management" => metric_id == "temperature_delta_c",
        "articulation" => matches!(
            metric_id,
            "joint_rom_delta_deg" | "flex_fatigue_cycles" | "minimum_clearance_mm"
        ),
        "abrasion" => metric_id == "abrasion_cycles",
        "fluid_chemical_barrier" => matches!(
            metric_id,
            "fluid_ingress_mass_g_m2" | "chemical_mass_change_pct"
        ),
        "cleanability" => metric_id == "cleaning_residue_fraction",
        "sensor_accommodation" => metric_id == "sensor_occlusion_fraction",
        "repairability" => metric_id == "repair_time_min",
        _ => false,
    }
}

fn bounds_are_consistent(requirement: &ZoneRequirement, frozen: bool) -> bool {
    if let (Some(lower), Some(upper)) = (requirement.lower_bound, requirement.upper_bound) {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return false;
        }
    }
    if requirement.lower_bound.is_some_and(|value| !value.is_finite())
        || requirement.upper_bound.is_some_and(|value| !value.is_finite())
    {
        return false;
    }

    match requirement.direction.as_str() {
        "higher_is_better" => {
            requirement.upper_bound.is_none() && (!frozen || requirement.lower_bound.is_some())
        }
        "lower_is_better" => {
            requirement.lower_bound.is_none() && (!frozen || requirement.upper_bound.is_some())
        }
        "within_declared_band" => {
            let both_or_neither =
                requirement.lower_bound.is_some() == requirement.upper_bound.is_some();
            both_or_neither && (!frozen || requirement.lower_bound.is_some())
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn proposed_manifest() -> Value {
        json!({
            "schema_version": "0.1.0",
            "spec_id": "LIGHTSHELL-ZONE-SPEC-SYNTHETIC-001",
            "state": "proposed",
            "platform": {
                "platform_id": "SYNTHETIC-ELBOW-001",
                "article_revision": null,
                "geometry_digest": null
            },
            "operating_domain": [],
            "zones": [{
                "zone_id": "elbow_gaiter",
                "robot_zone_ref": null,
                "frame_ref": null,
                "functions": ["articulation", "abrasion"],
                "requirements": [
                    {
                        "function_id": "articulation",
                        "metric_id": "joint_rom_delta_deg",
                        "unit": "deg",
                        "direction": "within_declared_band",
                        "hard_gate": true,
                        "lower_bound": null,
                        "upper_bound": null,
                        "method_ref": null,
                        "acceptance_rule_ref": null
                    },
                    {
                        "function_id": "articulation",
                        "metric_id": "flex_fatigue_cycles",
                        "unit": "cycles",
                        "direction": "higher_is_better",
                        "hard_gate": false,
                        "lower_bound": null,
                        "upper_bound": null,
                        "method_ref": null,
                        "acceptance_rule_ref": null
                    },
                    {
                        "function_id": "abrasion",
                        "metric_id": "abrasion_cycles",
                        "unit": "cycles",
                        "direction": "higher_is_better",
                        "hard_gate": false,
                        "lower_bound": null,
                        "upper_bound": null,
                        "method_ref": null,
                        "acceptance_rule_ref": null
                    }
                ],
                "keep_out_refs": [],
                "clearance_review_ref": null,
                "service": {
                    "replaceable": true,
                    "interface_ref": null,
                    "inspection_access": false
                }
            }],
            "evidence_policy": {
                "unknowns_fail_closed": true,
                "visual_estimates_are_non_qualifying": true,
                "simulation_is_not_physical_qualification": true,
                "supplier_claims_are_not_independent_qualification": true
            }
        })
    }

    #[test]
    fn proposed_zone_portfolio_is_admitted_but_not_frozen() {
        let input = proposed_manifest().to_string();
        let report = parse_and_validate(&input).expect("valid JSON");
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(!report.is_freezable());
        assert_eq!(report.zone_count, 1);
        assert_eq!(report.requirement_count, 3);
    }

    #[test]
    fn duplicate_zone_ids_are_rejected() {
        let mut manifest = proposed_manifest();
        let duplicate = manifest["zones"][0].clone();
        manifest["zones"].as_array_mut().unwrap().push(duplicate);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "duplicate_zone_id"));
    }

    #[test]
    fn unknown_function_is_rejected() {
        let mut manifest = proposed_manifest();
        manifest["zones"][0]["functions"][0] = json!("magic_absorption");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "unknown_function"));
    }

    #[test]
    fn function_without_compatible_measurement_is_rejected() {
        let mut manifest = proposed_manifest();
        manifest["zones"][0]["requirements"][0]["metric_id"] = json!("abrasion_cycles");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "function_metric_mismatch"));
        assert!(report.findings.iter().any(|item| item.code == "function_without_measurement"));
    }

    #[test]
    fn duplicate_function_metric_pair_is_rejected() {
        let mut manifest = proposed_manifest();
        let duplicate = manifest["zones"][0]["requirements"][0].clone();
        manifest["zones"][0]["requirements"].as_array_mut().unwrap().push(duplicate);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "duplicate_zone_requirement"));
    }

    #[test]
    fn contradictory_numeric_band_is_rejected() {
        let mut manifest = proposed_manifest();
        manifest["zones"][0]["requirements"][0]["lower_bound"] = json!(5.0);
        manifest["zones"][0]["requirements"][0]["upper_bound"] = json!(2.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "invalid_metric_bounds"));
    }

    #[test]
    fn noncanonical_units_are_rejected() {
        let mut manifest = proposed_manifest();
        manifest["zones"][0]["requirements"][0]["unit"] = json!("deg or rad");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "metric_unit_mismatch"));
    }

    #[test]
    fn frozen_spec_requires_exact_cad_and_method_acceptance_evidence() {
        let mut manifest = proposed_manifest();
        manifest["state"] = json!("frozen");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "frozen_platform_missing_revision"));
        assert!(report.findings.iter().any(|item| item.code == "frozen_platform_missing_geometry_digest"));
        assert!(report.findings.iter().any(|item| item.code == "frozen_zone_missing_cad_ref"));
        assert!(report.findings.iter().any(|item| item.code == "frozen_requirement_missing_method"));
        assert!(!report.is_freezable());
    }

    #[test]
    fn frozen_replaceable_zone_requires_service_interface() {
        let mut manifest = proposed_manifest();
        manifest["state"] = json!("frozen");
        manifest["platform"]["article_revision"] = json!("REV-A");
        manifest["platform"]["geometry_digest"] = json!(format!("sha256:{}", "a".repeat(64)));
        manifest["operating_domain"] = json!(["synthetic indoor low-energy motion"]);
        manifest["zones"][0]["robot_zone_ref"] = json!("CAD-ZONE-ELBOW-01");
        manifest["zones"][0]["frame_ref"] = json!("FRAME-ELBOW-BASE");
        manifest["zones"][0]["clearance_review_ref"] = json!("CLEARANCE-REVIEW-01");
        manifest["zones"][0]["service"]["interface_ref"] = json!("INTERFACE-ELBOW-01");
        manifest["zones"][0]["service"]["inspection_access"] = json!(true);
        for requirement in manifest["zones"][0]["requirements"].as_array_mut().unwrap() {
            let metric_id = requirement["metric_id"].as_str().unwrap();
            match metric_id {
                "joint_rom_delta_deg" => {
                    requirement["lower_bound"] = json!(-1.0);
                    requirement["upper_bound"] = json!(1.0);
                }
                "flex_fatigue_cycles" => requirement["lower_bound"] = json!(1000.0),
                "abrasion_cycles" => requirement["lower_bound"] = json!(100.0),
                _ => unreachable!(),
            }
            requirement["method_ref"] = json!("METHOD-REV-A");
            requirement["acceptance_rule_ref"] = json!("ACCEPTANCE-REV-A");
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_freezable(), "{:?}", report.findings);
    }

    #[test]
    fn frozen_replaceable_zone_without_interface_is_rejected() {
        let mut manifest = proposed_manifest();
        manifest["state"] = json!("frozen");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "replaceable_zone_not_serviceable"));
    }

    #[test]
    fn malformed_geometry_digest_is_rejected() {
        let mut manifest = proposed_manifest();
        manifest["platform"]["geometry_digest"] = json!("sha256:short");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "invalid_geometry_digest"));
    }

    #[test]
    fn unknown_fields_are_rejected_by_the_parser() {
        let mut manifest = proposed_manifest();
        manifest["unreviewed_property"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
