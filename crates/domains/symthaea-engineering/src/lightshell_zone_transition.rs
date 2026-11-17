// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Admission contracts for interfaces between functionally graded Lightshell zones.
//!
//! This module checks declared transition strategies, zone references, process
//! capability and required interface measurements. It does not predict textile
//! mechanics, authorize manufacturing, or qualify a physical garment.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionManifest {
    pub schema_version: String,
    pub program_id: String,
    pub state: String,
    pub zone_refs: Vec<ZoneRef>,
    pub process_capability: ProcessCapability,
    pub transitions: Vec<TransitionCandidate>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneRef {
    pub zone_id: String,
    pub functional_roles: Vec<String>,
    pub construction_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessCapability {
    pub capability_state: String,
    pub machine_ref: Option<String>,
    pub program_revision_ref: Option<String>,
    pub capability_evidence_ref: Option<String>,
    pub supported_strategies: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionCandidate {
    pub transition_id: String,
    pub from_zone_id: String,
    pub to_zone_id: String,
    pub strategy: String,
    pub state: String,
    pub transition_width_mm: Option<f64>,
    pub interface_detail_ref: Option<String>,
    pub process_route_ref: Option<String>,
    pub requirements: Vec<TransitionRequirement>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionRequirement {
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
pub struct EvidencePolicy {
    pub unknown_process_capability_fails_closed: bool,
    pub simulation_is_not_interface_qualification: bool,
    pub supplier_process_claims_are_not_capability_evidence: bool,
    pub coupon_interface_results_are_not_robot_qualification: bool,
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
    pub transition_count: usize,
    pub capability_checked_count: usize,
    pub transitions_frozen: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// A frozen transition manifest is a complete and capability-bound design
    /// contract, not physical performance or robot safety qualification.
    pub fn is_freezable(&self) -> bool {
        self.is_valid() && self.transitions_frozen
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: TransitionManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl TransitionManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut zones: HashMap<&str, &ZoneRef> = HashMap::new();
        let mut transition_ids = HashSet::new();
        let mut capability_checked_count = 0usize;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell transition schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.program_id) {
            push(&mut findings, "missing_program_id", "program_id",
                "Transition program identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "capability_pending" | "frozen") {
            push(&mut findings, "invalid_manifest_state", "state",
                "Manifest state must be proposed, capability_pending or frozen.");
        }
        if !matches!(
            self.process_capability.capability_state.as_str(),
            "unknown" | "preflight_pending" | "qualified_for_transition" | "not_capable"
        ) {
            push(&mut findings, "invalid_process_capability_state", "process_capability.capability_state",
                "Unknown manufacturing capability state.");
        }

        for (index, zone) in self.zone_refs.iter().enumerate() {
            let path = format!("zone_refs[{index}]");
            if is_blank(&zone.zone_id) {
                push(&mut findings, "blank_zone_id", format!("{path}.zone_id"),
                    "Zone reference identity must be non-empty.");
            }
            if zones.insert(zone.zone_id.as_str(), zone).is_some() {
                push(&mut findings, "duplicate_zone_ref", format!("{path}.zone_id"),
                    "Zone IDs must be unique within this transition manifest.");
            }
            if zone.functional_roles.is_empty() {
                push(&mut findings, "zone_without_functional_role", format!("{path}.functional_roles"),
                    "A transition endpoint must declare at least one functional role.");
            }
            let mut roles = HashSet::new();
            for (role_index, role) in zone.functional_roles.iter().enumerate() {
                if !is_known_function(role) {
                    push(&mut findings, "unknown_zone_function", format!("{path}.functional_roles[{role_index}]"),
                        "Unknown Lightshell functional role.");
                }
                if !roles.insert(role.as_str()) {
                    push(&mut findings, "duplicate_zone_function", format!("{path}.functional_roles[{role_index}]"),
                        "Functional roles must be unique within a zone reference.");
                }
            }
        }

        if zones.len() < 2 {
            push(&mut findings, "insufficient_zone_refs", "zone_refs",
                "At least two distinct zones are required for a transition program.");
        }
        if self.transitions.is_empty() {
            push(&mut findings, "no_transition_candidates", "transitions",
                "At least one transition candidate must be declared.");
        }

        for (index, transition) in self.transitions.iter().enumerate() {
            let path = format!("transitions[{index}]");
            if is_blank(&transition.transition_id) {
                push(&mut findings, "blank_transition_id", format!("{path}.transition_id"),
                    "Transition identity must be non-empty.");
            }
            if !transition_ids.insert(transition.transition_id.as_str()) {
                push(&mut findings, "duplicate_transition_id", format!("{path}.transition_id"),
                    "Transition IDs must be unique.");
            }
            if transition.from_zone_id == transition.to_zone_id {
                push(&mut findings, "self_transition", &path,
                    "A transition must connect two distinct zone IDs.");
            }
            if !zones.contains_key(transition.from_zone_id.as_str()) {
                push(&mut findings, "unknown_from_zone", format!("{path}.from_zone_id"),
                    "Source zone is absent from zone_refs.");
            }
            if !zones.contains_key(transition.to_zone_id.as_str()) {
                push(&mut findings, "unknown_to_zone", format!("{path}.to_zone_id"),
                    "Destination zone is absent from zone_refs.");
            }
            if !is_known_strategy(&transition.strategy) {
                push(&mut findings, "unknown_transition_strategy", format!("{path}.strategy"),
                    "Unknown textile transition strategy.");
            }
            if !matches!(transition.state.as_str(), "proposed" | "capability_checked" | "frozen") {
                push(&mut findings, "invalid_transition_state", format!("{path}.state"),
                    "Transition state must be proposed, capability_checked or frozen.");
            }
            if transition.transition_width_mm.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                push(&mut findings, "invalid_transition_width", format!("{path}.transition_width_mm"),
                    "A supplied transition width must be finite and positive.");
            }

            if transition.state == "capability_checked" || transition.state == "frozen" {
                capability_checked_count += 1;
                if self.process_capability.capability_state != "qualified_for_transition" {
                    push(&mut findings, "transition_without_process_capability", format!("{path}.state"),
                        "A transition cannot be capability_checked or frozen before its manufacturing process is qualified.");
                }
                if !self.process_capability.supported_strategies.contains(&transition.strategy) {
                    push(&mut findings, "strategy_not_in_capability_scope", format!("{path}.strategy"),
                        "The declared process capability does not include this transition strategy.");
                }
            }

            let mut metrics = HashSet::new();
            for (metric_index, requirement) in transition.requirements.iter().enumerate() {
                let mpath = format!("{path}.requirements[{metric_index}]");
                if !metrics.insert(requirement.metric_id.as_str()) {
                    push(&mut findings, "duplicate_transition_metric", format!("{mpath}.metric_id"),
                        "Transition metric IDs must be unique within a candidate.");
                }
                if !metric_supported_by_strategy(&transition.strategy, &requirement.metric_id) {
                    push(&mut findings, "unsupported_transition_metric", format!("{mpath}.metric_id"),
                        "Metric is not part of the declared transition strategy's evidence contract.");
                }
                if !unit_matches_metric(&requirement.metric_id, &requirement.unit) {
                    push(&mut findings, "transition_metric_unit_mismatch", format!("{mpath}.unit"),
                        "Metric unit does not match the canonical unit.");
                }
                if direction_for_metric(&requirement.metric_id) != Some(requirement.direction.as_str()) {
                    push(&mut findings, "transition_metric_direction_mismatch", format!("{mpath}.direction"),
                        "Metric comparison direction does not match its semantics.");
                }
                if !bounds_are_consistent(requirement) {
                    push(&mut findings, "invalid_transition_bounds", format!("{mpath}.lower_bound/upper_bound"),
                        "Metric bounds are contradictory or inconsistent with comparison direction.");
                }
                if transition.state == "frozen" {
                    if requirement.lower_bound.is_none() && requirement.upper_bound.is_none() {
                        push(&mut findings, "frozen_transition_missing_bound", &mpath,
                            "A frozen transition requirement requires an explicit acceptance bound.");
                    }
                    if is_blank_option(&requirement.method_ref) {
                        push(&mut findings, "frozen_transition_missing_method", format!("{mpath}.method_ref"),
                            "A frozen transition requirement must cite a measurement method.");
                    }
                    if is_blank_option(&requirement.acceptance_rule_ref) {
                        push(&mut findings, "frozen_transition_missing_acceptance", format!("{mpath}.acceptance_rule_ref"),
                            "A frozen transition requirement must cite an acceptance rule.");
                    }
                }
            }

            if is_known_strategy(&transition.strategy) {
                for expected_metric in required_metrics(&transition.strategy) {
                    if !metrics.contains(expected_metric) {
                        push(&mut findings, "missing_strategy_requirement", format!("{path}.requirements"),
                            format!("Strategy {} requires metric {}.", transition.strategy, expected_metric));
                    }
                }
            }

            if transition.state == "capability_checked" || transition.state == "frozen" {
                for (field, value) in [
                    ("interface_detail_ref", transition.interface_detail_ref.as_deref()),
                    ("process_route_ref", transition.process_route_ref.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "capability_checked_transition_missing_detail", format!("{path}.{field}"),
                            "A capability-checked transition must bind its interface detail and process route.");
                    }
                }
            }

            if transition.state == "frozen" {
                for zone_id in [&transition.from_zone_id, &transition.to_zone_id] {
                    if let Some(zone) = zones.get(zone_id.as_str()) {
                        if is_blank_option(&zone.construction_ref) {
                            push(&mut findings, "frozen_transition_zone_missing_construction", format!("{path}.zone_refs[{zone_id}].construction_ref"),
                                "A frozen transition must bind the exact construction on both sides of the interface.");
                        }
                    }
                }
                for (field, value) in [
                    ("transition_width_mm", transition.transition_width_mm.map(|_| "present")),
                    ("interface_detail_ref", transition.interface_detail_ref.as_deref()),
                    ("process_route_ref", transition.process_route_ref.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "frozen_transition_missing_process_detail", format!("{path}.{field}"),
                            "A frozen transition must bind its width, interface detail and manufacturing route.");
                    }
                }
            }
        }

        let mut supported_strategies = HashSet::new();
        for (index, strategy) in self.process_capability.supported_strategies.iter().enumerate() {
            if !is_known_strategy(strategy) {
                push(&mut findings, "unknown_capability_strategy", format!("process_capability.supported_strategies[{index}]"),
                    "Capability evidence must name a known transition strategy.");
            }
            if !supported_strategies.insert(strategy.as_str()) {
                push(&mut findings, "duplicate_capability_strategy", format!("process_capability.supported_strategies[{index}]"),
                    "Supported transition strategies must be unique.");
            }
        }
        if self.process_capability.capability_state == "qualified_for_transition" {
            for (field, value) in [
                ("machine_ref", self.process_capability.machine_ref.as_deref()),
                ("program_revision_ref", self.process_capability.program_revision_ref.as_deref()),
                ("capability_evidence_ref", self.process_capability.capability_evidence_ref.as_deref()),
            ] {
                if value.is_none_or(is_blank) {
                    push(&mut findings, "qualified_process_missing_identity", format!("process_capability.{field}"),
                        "Qualified process capability requires exact machine, program revision and capability evidence.");
                }
            }
            if self.process_capability.supported_strategies.is_empty() {
                push(&mut findings, "qualified_process_without_strategies", "process_capability.supported_strategies",
                    "A qualified process must declare the supported transition strategies.");
            }
        } else if !self.process_capability.supported_strategies.is_empty() {
            push(&mut findings, "unqualified_process_claims_supported_strategies", "process_capability.supported_strategies",
                "Only a qualified process may list supported transition strategies.");
        }

        if self.state == "frozen" {
            if self.process_capability.capability_state != "qualified_for_transition" {
                push(&mut findings, "frozen_manifest_without_capability", "process_capability.capability_state",
                    "A frozen transition manifest requires qualified manufacturing capability.");
            }
            if self.transitions.iter().any(|transition| transition.state != "frozen") {
                push(&mut findings, "frozen_manifest_has_unfrozen_transition", "transitions",
                    "Every transition must be frozen before the manifest can freeze.");
            }
        }

        if !self.evidence_policy.unknown_process_capability_fails_closed
            || !self.evidence_policy.simulation_is_not_interface_qualification
            || !self.evidence_policy.supplier_process_claims_are_not_capability_evidence
            || !self.evidence_policy.coupon_interface_results_are_not_robot_qualification
        {
            push(&mut findings, "unsafe_transition_evidence_policy", "evidence_policy",
                "Process unknowns, simulation and supplier claims must not be promoted to physical interface qualification.");
        }

        ValidationReport {
            findings,
            zone_count: self.zone_refs.len(),
            transition_count: self.transitions.len(),
            capability_checked_count,
            transitions_frozen: self.state == "frozen",
        }
    }
}

fn required_metrics(strategy: &str) -> &'static [&'static str] {
    match strategy {
        "graded_knit" => &[
            "transition_strain_concentration_ratio",
            "stiffness_jump_ratio",
            "compression_continuity_ratio",
        ],
        "intarsia" => &[
            "transition_strain_concentration_ratio",
            "stiffness_jump_ratio",
            "delamination_cycles",
        ],
        "sewn_seam" => &[
            "seam_strength_n",
            "seam_fatigue_cycles",
            "compression_continuity_ratio",
        ],
        "bonded_laminate" => &[
            "peel_strength_n_per_mm",
            "delamination_cycles",
            "flex_fatigue_cycles",
        ],
        "printed_reinforcement" => &[
            "peel_strength_n_per_mm",
            "delamination_cycles",
            "added_mass_g",
        ],
        "replaceable_overlap" => &[
            "attachment_retention_cycles",
            "repair_time_min",
            "added_mass_g",
        ],
        "spacer_stitch" => &[
            "compression_continuity_ratio",
            "stiffness_jump_ratio",
            "transition_strain_concentration_ratio",
        ],
        _ => &[],
    }
}

fn metric_supported_by_strategy(strategy: &str, metric: &str) -> bool {
    if required_metrics(strategy).contains(&metric) {
        return true;
    }
    match strategy {
        "graded_knit" | "intarsia" | "spacer_stitch" => {
            matches!(metric, "air_permeability_ratio" | "thermal_delta_c" | "repair_time_min" | "added_mass_g")
        }
        "sewn_seam" => {
            matches!(metric, "delamination_cycles" | "flex_fatigue_cycles" | "air_permeability_ratio" | "repair_time_min" | "added_mass_g")
        }
        "bonded_laminate" => {
            matches!(metric, "seam_fatigue_cycles" | "air_permeability_ratio" | "thermal_delta_c" | "repair_time_min" | "added_mass_g")
        }
        "printed_reinforcement" => {
            matches!(metric, "transition_strain_concentration_ratio" | "stiffness_jump_ratio" | "compression_continuity_ratio" | "repair_time_min")
        }
        "replaceable_overlap" => {
            matches!(metric, "seam_strength_n" | "seam_fatigue_cycles" | "delamination_cycles" | "air_permeability_ratio" | "thermal_delta_c")
        }
        _ => false,
    }
}

fn is_known_strategy(value: &str) -> bool {
    matches!(value,
        "graded_knit" | "intarsia" | "sewn_seam" | "bonded_laminate" |
        "printed_reinforcement" | "replaceable_overlap" | "spacer_stitch")
}

fn is_known_function(value: &str) -> bool {
    matches!(value,
        "impact_compression" | "vibration_isolation" | "airflow" |
        "thermal_management" | "articulation" | "abrasion" |
        "fluid_chemical_barrier" | "cleanability" | "sensor_accommodation" |
        "repairability" | "mass_budget" | "seam_integrity")
}

fn metric_definition(value: &str) -> Option<(&'static str, &'static str)> {
    match value {
        "transition_strain_concentration_ratio" => Some(("ratio", "lower_is_better")),
        "stiffness_jump_ratio" => Some(("ratio", "within_declared_band")),
        "compression_continuity_ratio" => Some(("ratio", "within_declared_band")),
        "seam_strength_n" => Some(("N", "higher_is_better")),
        "seam_fatigue_cycles" => Some(("cycles", "higher_is_better")),
        "peel_strength_n_per_mm" => Some(("N/mm", "higher_is_better")),
        "delamination_cycles" => Some(("cycles", "higher_is_better")),
        "flex_fatigue_cycles" => Some(("cycles", "higher_is_better")),
        "added_mass_g" => Some(("g", "lower_is_better")),
        "attachment_retention_cycles" => Some(("cycles", "higher_is_better")),
        "repair_time_min" => Some(("min", "lower_is_better")),
        "air_permeability_ratio" => Some(("ratio", "within_declared_band")),
        "thermal_delta_c" => Some(("degC", "within_declared_band")),
        _ => None,
    }
}

fn unit_matches_metric(metric: &str, unit: &str) -> bool {
    metric_definition(metric).is_some_and(|(canonical, _)| canonical == unit)
}

fn direction_for_metric(metric: &str) -> Option<&'static str> {
    metric_definition(metric).map(|(_, direction)| direction)
}

fn bounds_are_consistent(requirement: &TransitionRequirement) -> bool {
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
        "higher_is_better" => requirement.upper_bound.is_none(),
        "lower_is_better" => requirement.lower_bound.is_none(),
        "within_declared_band" => {
            requirement.lower_bound.is_some() == requirement.upper_bound.is_some()
        }
        _ => false,
    }
}

fn push(findings: &mut Vec<Finding>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_blank_option(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_blank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn base_manifest() -> Value {
        json!({
            "schema_version": "0.1.0",
            "program_id": "LIGHTSHELL-TRANSITION-SYNTHETIC-001",
            "state": "proposed",
            "zone_refs": [
                {
                    "zone_id": "impact_panel",
                    "functional_roles": ["impact_compression"],
                    "construction_ref": null
                },
                {
                    "zone_id": "articulation_gaiter",
                    "functional_roles": ["articulation"],
                    "construction_ref": null
                },
                {
                    "zone_id": "vent_panel",
                    "functional_roles": ["airflow"],
                    "construction_ref": null
                },
                {
                    "zone_id": "sensor_insert",
                    "functional_roles": ["sensor_accommodation"],
                    "construction_ref": null
                },
                {
                    "zone_id": "passive_textile",
                    "functional_roles": ["repairability"],
                    "construction_ref": null
                }
            ],
            "process_capability": {
                "capability_state": "unknown",
                "machine_ref": null,
                "program_revision_ref": null,
                "capability_evidence_ref": null,
                "supported_strategies": []
            },
            "transitions": [
                {
                    "transition_id": "impact_to_gaiter",
                    "from_zone_id": "impact_panel",
                    "to_zone_id": "articulation_gaiter",
                    "strategy": "graded_knit",
                    "state": "proposed",
                    "transition_width_mm": null,
                    "interface_detail_ref": null,
                    "process_route_ref": null,
                    "requirements": [
                        {"metric_id":"transition_strain_concentration_ratio","unit":"ratio","direction":"lower_is_better","hard_gate":true,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null},
                        {"metric_id":"stiffness_jump_ratio","unit":"ratio","direction":"within_declared_band","hard_gate":true,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null},
                        {"metric_id":"compression_continuity_ratio","unit":"ratio","direction":"within_declared_band","hard_gate":true,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null}
                    ]
                },
                {
                    "transition_id": "vent_to_service",
                    "from_zone_id": "vent_panel",
                    "to_zone_id": "passive_textile",
                    "strategy": "sewn_seam",
                    "state": "proposed",
                    "transition_width_mm": null,
                    "interface_detail_ref": null,
                    "process_route_ref": null,
                    "requirements": [
                        {"metric_id":"seam_strength_n","unit":"N","direction":"higher_is_better","hard_gate":true,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null},
                        {"metric_id":"seam_fatigue_cycles","unit":"cycles","direction":"higher_is_better","hard_gate":false,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null},
                        {"metric_id":"compression_continuity_ratio","unit":"ratio","direction":"within_declared_band","hard_gate":false,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null}
                    ]
                },
                {
                    "transition_id": "sensor_to_passive",
                    "from_zone_id": "sensor_insert",
                    "to_zone_id": "passive_textile",
                    "strategy": "replaceable_overlap",
                    "state": "proposed",
                    "transition_width_mm": null,
                    "interface_detail_ref": null,
                    "process_route_ref": null,
                    "requirements": [
                        {"metric_id":"attachment_retention_cycles","unit":"cycles","direction":"higher_is_better","hard_gate":true,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null},
                        {"metric_id":"repair_time_min","unit":"min","direction":"lower_is_better","hard_gate":false,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null},
                        {"metric_id":"added_mass_g","unit":"g","direction":"lower_is_better","hard_gate":false,"lower_bound":null,"upper_bound":null,"method_ref":null,"acceptance_rule_ref":null}
                    ]
                }
            ],
            "evidence_policy": {
                "unknown_process_capability_fails_closed": true,
                "simulation_is_not_interface_qualification": true,
                "supplier_process_claims_are_not_capability_evidence": true,
                "coupon_interface_results_are_not_robot_qualification": true
            }
        })
    }

    #[test]
    fn proposed_transition_portfolio_is_admitted_without_claiming_capability() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.zone_count, 5);
        assert_eq!(report.transition_count, 3);
        assert!(!report.is_freezable());
    }

    #[test]
    fn duplicate_zone_and_transition_ids_are_rejected() {
        let mut manifest = base_manifest();
        let zone = manifest["zone_refs"][0].clone();
        manifest["zone_refs"].as_array_mut().unwrap().push(zone);
        let transition = manifest["transitions"][0].clone();
        manifest["transitions"].as_array_mut().unwrap().push(transition);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_zone_ref"));
        assert!(report.findings.iter().any(|f| f.code == "duplicate_transition_id"));
    }

    #[test]
    fn transition_requires_distinct_declared_zone_refs() {
        let mut manifest = base_manifest();
        manifest["transitions"][0]["to_zone_id"] = json!("unknown_zone");
        manifest["transitions"][1]["to_zone_id"] = json!("vent_panel");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unknown_to_zone"));
        assert!(report.findings.iter().any(|f| f.code == "self_transition"));
    }

    #[test]
    fn unsupported_strategy_is_rejected() {
        let mut manifest = base_manifest();
        manifest["transitions"][0]["strategy"] = json!("magic_seam");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unknown_transition_strategy"));
    }

    #[test]
    fn strategy_specific_measurements_are_mandatory() {
        let mut manifest = base_manifest();
        manifest["transitions"][0]["requirements"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "missing_strategy_requirement"));
    }

    #[test]
    fn transition_metric_units_and_directions_are_canonical() {
        let mut manifest = base_manifest();
        manifest["transitions"][0]["requirements"][0]["unit"] = json!("N");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "transition_metric_unit_mismatch"));
    }

    #[test]
    fn contradictory_metric_bounds_are_rejected() {
        let mut manifest = base_manifest();
        manifest["transitions"][0]["requirements"][1]["lower_bound"] = json!(2.0);
        manifest["transitions"][0]["requirements"][1]["upper_bound"] = json!(1.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "invalid_transition_bounds"));
    }

    #[test]
    fn frozen_manifest_requires_qualified_process_and_frozen_transitions() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("frozen");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "frozen_manifest_without_capability"));
        assert!(report.findings.iter().any(|f| f.code == "frozen_manifest_has_unfrozen_transition"));
    }

    #[test]
    fn capability_checked_transition_requires_explicit_strategy_support() {
        let mut manifest = base_manifest();
        manifest["transitions"][0]["state"] = json!("capability_checked");
        manifest["process_capability"]["capability_state"] = json!("qualified_for_transition");
        manifest["process_capability"]["machine_ref"] = json!("MACHINE-001");
        manifest["process_capability"]["program_revision_ref"] = json!("PROGRAM-REV-A");
        manifest["process_capability"]["capability_evidence_ref"] = json!("CAPABILITY-TEST-001");
        manifest["process_capability"]["supported_strategies"] = json!(["sewn_seam"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "strategy_not_in_capability_scope"));
    }

    #[test]
    fn frozen_transition_requires_width_process_route_and_method_acceptance() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("frozen");
        manifest["process_capability"]["capability_state"] = json!("qualified_for_transition");
        manifest["process_capability"]["machine_ref"] = json!("MACHINE-001");
        manifest["process_capability"]["program_revision_ref"] = json!("PROGRAM-REV-A");
        manifest["process_capability"]["capability_evidence_ref"] = json!("CAPABILITY-TEST-001");
        manifest["process_capability"]["supported_strategies"] = json!(["graded_knit","sewn_seam","replaceable_overlap"]);
        for transition in manifest["transitions"].as_array_mut().unwrap() {
            transition["state"] = json!("frozen");
            transition["transition_width_mm"] = json!(12.0);
            transition["interface_detail_ref"] = json!("INTERFACE-DETAIL-REV-A");
            transition["process_route_ref"] = json!("PROCESS-ROUTE-REV-A");
            for requirement in transition["requirements"].as_array_mut().unwrap() {
                match requirement["direction"].as_str().unwrap() {
                    "higher_is_better" => requirement["lower_bound"] = json!(1.0),
                    "lower_is_better" => requirement["upper_bound"] = json!(10.0),
                    "within_declared_band" => {
                        requirement["lower_bound"] = json!(0.8);
                        requirement["upper_bound"] = json!(1.2);
                    }
                    _ => unreachable!()
                }
                requirement["method_ref"] = json!("METHOD-REV-A");
                requirement["acceptance_rule_ref"] = json!("ACCEPTANCE-REV-A");
            }
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_freezable(), "{:?}", report.findings);
    }

    #[test]
    fn unqualified_process_cannot_claim_supported_strategies() {
        let mut manifest = base_manifest();
        manifest["process_capability"]["supported_strategies"] = json!(["graded_knit"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unqualified_process_claims_supported_strategies"));
    }

    #[test]
    fn frozen_transition_requires_construction_identity_for_both_zones() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("frozen");
        manifest["process_capability"]["capability_state"] = json!("qualified_for_transition");
        manifest["process_capability"]["machine_ref"] = json!("MACHINE-001");
        manifest["process_capability"]["program_revision_ref"] = json!("PROGRAM-REV-A");
        manifest["process_capability"]["capability_evidence_ref"] = json!("CAPABILITY-TEST-001");
        manifest["process_capability"]["supported_strategies"] = json!(["graded_knit","sewn_seam","replaceable_overlap"]);
        for zone in manifest["zone_refs"].as_array_mut().unwrap() {
            zone["construction_ref"] = json!("CONSTRUCTION-REV-A");
        }
        for transition in manifest["transitions"].as_array_mut().unwrap() {
            transition["state"] = json!("frozen");
            transition["transition_width_mm"] = json!(12.0);
            transition["interface_detail_ref"] = json!("INTERFACE-DETAIL-REV-A");
            transition["process_route_ref"] = json!("PROCESS-ROUTE-REV-A");
            for requirement in transition["requirements"].as_array_mut().unwrap() {
                match requirement["direction"].as_str().unwrap() {
                    "higher_is_better" => requirement["lower_bound"] = json!(1.0),
                    "lower_is_better" => requirement["upper_bound"] = json!(10.0),
                    "within_declared_band" => {
                        requirement["lower_bound"] = json!(0.8);
                        requirement["upper_bound"] = json!(1.2);
                    }
                    _ => unreachable!()
                }
                requirement["method_ref"] = json!("METHOD-REV-A");
                requirement["acceptance_rule_ref"] = json!("ACCEPTANCE-REV-A");
            }
        }
        manifest["zone_refs"][0]["construction_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "frozen_transition_zone_missing_construction"));
    }

    #[test]
    fn qualified_process_requires_machine_program_and_evidence() {
        let mut manifest = base_manifest();
        manifest["process_capability"]["capability_state"] = json!("qualified_for_transition");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "qualified_process_missing_identity"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["unreviewed_process_claim"] = json!("production ready");
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
