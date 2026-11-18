// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Application-specific cleaning and post-clean functional qualification.
//!
//! Composes Lightshell coupon, method-applicability, mission-exposure,
//! installed-system and lifecycle owners. This module admits only evidence
//! about a declared soil challenge and the exact cleaning process/article.
//! It does not certify hygiene, sterility, chemical safety or robot operation.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanStateManifest {
    pub schema_version: String,
    pub campaign_id: String,
    pub state: String,
    pub frozen_before_first_cycle: bool,
    pub preregistration_ref: String,
    pub protocol_digest: String,
    pub required_cycle_count: u32,
    pub target_trial_design_ref: Option<String>,
    pub article: ArticleBinding,
    pub application: ApplicationProfile,
    pub soil_challenge: SoilChallenge,
    pub gates: Vec<ClaimGate>,
    pub cycles: Vec<CleaningCycle>,
    pub independent_review: Option<IndependentReview>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArticleBinding {
    pub candidate_id: String,
    pub garment_generation_ref: String,
    pub manufactured_article_ref: String,
    pub installed_article_digest: String,
    pub robot_id: String,
    pub robot_serial: String,
    pub firmware_revision: String,
    pub robot_configuration_digest: String,
    pub mission_profile_ref: String,
    pub topology_digest: String,
    pub asbuilt_inspection_ref: String,
    pub cleaning_mode: String,
    pub zones: Vec<ZoneBinding>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneBinding {
    pub zone_ref: String,
    pub panel_ref: String,
    pub material_lot_ref: String,
    pub construction_ref: String,
    pub cleanability_sample_location_refs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationProfile {
    pub application_ref: String,
    pub cleaning_requirement_ref: String,
    pub acceptance_owner_ref: String,
    pub use_environment_ref: String,
    pub cleaning_interval_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SoilChallenge {
    pub soil_profile_ref: String,
    pub soil_class: String,
    pub soil_batch_ref: String,
    pub challenge_recipe_ref: String,
    pub application_method_ref: String,
    pub challenge_area_cm2: f64,
    pub target_loading: f64,
    pub loading_uncertainty: f64,
    pub loading_unit: String,
    pub dwell_seconds: f64,
    pub conditioning_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimGate {
    pub gate_id: String,
    pub zone_ref: String,
    pub required_sample_location_refs: Vec<String>,
    pub metric_family: String,
    pub comparison_mode: String,
    pub unit: String,
    pub acceptance_limit: f64,
    pub hard_gate: bool,
    pub applicability: String,
    pub applicability_owner_ref: Option<String>,
    pub not_applicable_rationale_ref: Option<String>,
    pub minimum_replicates_per_cycle: u32,
    pub method_selection_ref: String,
    pub method_applicability_ref: String,
    pub acceptance_rule_ref: String,
    pub assay_controls: Option<AssayControls>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssayControls {
    pub blank_control_ref: String,
    pub blank_response: f64,
    pub blank_uncertainty: f64,
    pub blank_maximum: f64,
    pub positive_control_ref: String,
    pub positive_control_response: f64,
    pub positive_control_uncertainty: f64,
    pub positive_control_minimum: f64,
    pub recovery_control_ref: String,
    pub recovery_fraction: f64,
    pub recovery_uncertainty: f64,
    pub recovery_minimum: f64,
    pub recovery_maximum: f64,
    pub controls_raw_data_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleaningCycle {
    pub cycle_id: String,
    pub cycle_index: u32,
    pub cleaning_method_selection_ref: String,
    pub cleaning_method_applicability_ref: String,
    pub mission_exposure_plan_ref: String,
    pub cleaning_mode: String,
    pub cleaning_agent_ref: String,
    pub cleaning_agent_batch_ref: String,
    pub agent_concentration: f64,
    pub agent_concentration_unit: String,
    pub contact_time_seconds: f64,
    pub surface_temperature_c: f64,
    pub application_mode: String,
    pub application_pressure_kpa: Option<f64>,
    pub nozzle_ref: Option<String>,
    pub standoff_cm: Option<f64>,
    pub rinse_water_quality_ref: String,
    pub rinse_volume_l: f64,
    pub drying_method_ref: String,
    pub robot_isolation_evidence_ref: Option<String>,
    pub removal_record_ref: Option<String>,
    pub reinstallation_inspection_ref: Option<String>,
    pub soil_application_ref: Option<String>,
    pub cleaning_run_ref: Option<String>,
    pub process_record_digest: Option<String>,
    pub raw_process_digest: Option<String>,
    pub post_clean_inspection: Option<Inspection>,
    pub observations: Vec<ClaimObservation>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inspection {
    pub disposition: String,
    pub inspector_ref: String,
    pub evidence_ref: String,
    pub defect_codes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimObservation {
    pub gate_id: String,
    pub zone_ref: String,
    pub sample_location_ref: String,
    pub replicate_index: u32,
    pub before: Option<Measurement>,
    pub after: Measurement,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub value: f64,
    pub expanded_uncertainty: f64,
    pub unit: String,
    pub quantification_state: String,
    pub detection_limit: f64,
    pub quantification_limit: f64,
    pub instrument_ref: String,
    pub calibration_ref: String,
    pub calibration_state: String,
    pub method_selection_ref: String,
    pub raw_data_digest: String,
    pub evidence_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentReview {
    pub disposition: String,
    pub reviewer_ref: String,
    pub reviewed_at: String,
    pub review_record_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateAssessment {
    pub gate_id: String,
    pub zone_ref: String,
    pub disposition: String,
    pub observation_count: usize,
    pub passing_observation_count: usize,
    pub worst_case_bound: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub cycle_count: usize,
    pub observation_count: usize,
    pub passed_gate_count: usize,
    pub failed_gate_count: usize,
    pub uncertainty_blocked_gate_count: usize,
    pub insufficient_evidence_gate_count: usize,
    pub not_applicable_gate_count: usize,
    pub gate_assessments: Vec<GateAssessment>,
    pub cleaning_qualification_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// This is limited to the exact article, zone, soil, cleaning process,
    /// method and cycle count in this manifest. It is not a hygiene certificate.
    pub fn can_admit_cleaning_qualification(&self) -> bool {
        self.is_valid() && self.cleaning_qualification_admitted
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: CleanStateManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl CleanStateManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut gate_map: HashMap<(&str, &str), &ClaimGate> = HashMap::new();
        let mut cycle_ids = HashSet::new();
        let mut cleaning_run_ids = HashSet::new();
        let mut soil_application_ids = HashSet::new();
        let mut observation_counts: HashMap<(&str, &str, &str, u32), usize> = HashMap::new();
        let mut observation_ids = HashSet::new();
        let mut assessments = Vec::new();
        let mut total_observations = 0usize;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only clean-state schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.campaign_id) {
            push(&mut findings, "campaign_identity_missing", "campaign_id",
                "Cleaning qualification campaign identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "measured" | "adjudicated") {
            push(&mut findings, "invalid_campaign_state", "state",
                "Campaign state must be proposed, measured or adjudicated.");
        }
        if !self.frozen_before_first_cycle || is_blank(&self.preregistration_ref)
            || !is_sha256(&self.protocol_digest)
        {
            push(&mut findings, "protocol_not_frozen", "preregistration_ref",
                "A pre-cycle frozen protocol and SHA-256 digest are required.");
        }
        if self.required_cycle_count == 0 || self.cycles.len() != self.required_cycle_count as usize {
            push(&mut findings, "required_cycle_count_mismatch", "required_cycle_count",
                "The observed/planned cycle list must exactly match the pre-registered required cycle count.");
        }
        if self.target_trial_design_ref.as_deref().is_some_and(is_blank) {
            push(&mut findings, "blank_trial_design_reference", "target_trial_design_ref",
                "Optional trial-design reference must be non-empty when supplied.");
        }

        validate_article(&self.article, self.state.as_str(), &mut findings);
        validate_application(&self.application, &mut findings);
        validate_soil_challenge(&self.soil_challenge, &mut findings);

        if self.gates.is_empty() {
            push(&mut findings, "campaign_without_claim_gates", "gates",
                "At least one applicable or explicitly not-applicable claim gate is required.");
        }
        for (index, gate) in self.gates.iter().enumerate() {
            let path = format!("gates[{index}]");
            if is_blank(&gate.gate_id) || is_blank(&gate.zone_ref)
                || gate_map.insert((gate.gate_id.as_str(), gate.zone_ref.as_str()), gate).is_some()
            {
                push(&mut findings, "duplicate_or_blank_gate_identity", &path,
                    "Gate identity must be a unique non-empty gate ID and zone pair.");
            }
            if !self.article.zones.iter().any(|zone| zone.zone_ref == gate.zone_ref) {
                push(&mut findings, "gate_zone_not_in_article", format!("{path}.zone_ref"),
                    "Claim gate refers to a zone that is not bound to the exact article.");
            }
            let zone_binding = self.article.zones.iter().find(|zone| zone.zone_ref == gate.zone_ref);
            let mut sample_locations = HashSet::new();
            if gate.required_sample_location_refs.is_empty()
                || gate.required_sample_location_refs.iter().any(|location| {
                    is_blank(location)
                        || !sample_locations.insert(location.as_str())
                        || zone_binding.is_none_or(|zone| !zone.cleanability_sample_location_refs.contains(location))
                })
            {
                push(&mut findings, "gate_sample_locations_not_topology_bound",
                    format!("{path}.required_sample_location_refs"),
                    "Every gate requires unique, non-empty sampling locations linked to the exact zone topology.");
            }
            if !metric_mode_is_compatible(&gate.metric_family, &gate.comparison_mode) {
                push(&mut findings, "unsupported_metric_comparison", &path,
                    "Metric family and comparison mode are incompatible or outside the P0 vocabulary.");
            }
            if !gate.hard_gate {
                push(&mut findings, "p0_gate_must_be_noncompensable", format!("{path}.hard_gate"),
                    "Every P0 required claim gate is non-compensable; use a separately scoped informational observation for non-gating metrics.");
            }
            if is_blank(&gate.unit) || !gate.acceptance_limit.is_finite()
                || gate.acceptance_limit < 0.0 || gate.minimum_replicates_per_cycle == 0
                || is_blank(&gate.method_selection_ref)
                || is_blank(&gate.method_applicability_ref)
                || is_blank(&gate.acceptance_rule_ref)
            {
                push(&mut findings, "invalid_gate_definition", &path,
                    "Gate requires a unit, finite non-negative limit, replicate minimum, method, applicability and acceptance-rule references.");
            }
            if gate.comparison_mode == "max_residual_fraction" && gate.acceptance_limit > 1.0 {
                push(&mut findings, "residual_fraction_limit_out_of_range",
                    format!("{path}.acceptance_limit"),
                    "Residual fraction upper limit must be between zero and one.");
            }
            if gate.comparison_mode == "min_retention_fraction" && gate.acceptance_limit > 1.0 {
                push(&mut findings, "retention_fraction_limit_out_of_range",
                    format!("{path}.acceptance_limit"),
                    "Retention fraction lower limit must be between zero and one.");
            }
            match gate.applicability.as_str() {
                "required" => {
                    if gate.applicability_owner_ref.is_some()
                        || gate.not_applicable_rationale_ref.is_some()
                    {
                        push(&mut findings, "required_gate_has_na_rationale", &path,
                            "Required gates must not carry a not-applicable disposition.");
                    }
                }
                "not_applicable" => {
                    if gate.applicability_owner_ref.as_deref().is_none_or(is_blank)
                        || gate.not_applicable_rationale_ref.as_deref().is_none_or(is_blank)
                    {
                        push(&mut findings, "na_gate_without_owner_rationale", &path,
                            "Not-applicable gates require a named owner and rationale.");
                    }
                }
                _ => push(&mut findings, "invalid_gate_applicability",
                    format!("{path}.applicability"),
                    "Applicability must be required or not_applicable."),
            }
            let assay_required = matches!(gate.metric_family.as_str(), "soil_residue" | "cleaner_residue");
            if assay_required && gate.assay_controls.is_none() {
                push(&mut findings, "assay_controls_missing", format!("{path}.assay_controls"),
                    "Soil and cleaner-residue gates require blank, positive and recovery controls.");
            }
            if let Some(controls) = gate.assay_controls.as_ref() {
                validate_assay_controls(controls, &path, &mut findings);
            }
        }

        if self.cycles.is_empty() {
            push(&mut findings, "campaign_without_cycles", "cycles",
                "At least one planned or executed cleaning cycle is required.");
        }

        for (cycle_index, cycle) in self.cycles.iter().enumerate() {
            let path = format!("cycles[{cycle_index}]");
            if is_blank(&cycle.cycle_id) || !cycle_ids.insert(cycle.cycle_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_cycle_id", format!("{path}.cycle_id"),
                    "Cleaning cycle IDs must be non-empty and unique.");
            }
            if cycle.cycle_index as usize != cycle_index {
                push(&mut findings, "cycle_index_not_contiguous", format!("{path}.cycle_index"),
                    "Cycle indices must be contiguous and start at zero.");
            }
            validate_cleaning_process(cycle, &self.article, self.state.as_str(), &path, &mut findings);

            if self.state == "proposed" {
                if cycle.soil_application_ref.is_some()
                    || cycle.cleaning_run_ref.is_some()
                    || cycle.process_record_digest.is_some()
                    || cycle.raw_process_digest.is_some()
                    || cycle.post_clean_inspection.is_some()
                    || !cycle.observations.is_empty()
                {
                    push(&mut findings, "proposed_cycle_contains_observations", &path,
                        "Proposed cycles cannot contain executed process records, inspections or measurements.");
                }
            } else {
                if cycle.soil_application_ref.as_deref().is_none_or(is_blank)
                    || cycle.cleaning_run_ref.as_deref().is_none_or(is_blank)
                    || cycle.process_record_digest.as_deref().is_none_or(|v| !is_sha256(v))
                    || cycle.raw_process_digest.as_deref().is_none_or(|v| !is_sha256(v))
                {
                    push(&mut findings, "cycle_execution_provenance_missing", &path,
                        "Measured cycles require soil application, cleaning run, process receipt and raw-process SHA-256.");
                }
                if let Some(run_ref) = cycle.cleaning_run_ref.as_deref() {
                    if !cleaning_run_ids.insert(run_ref) {
                        push(&mut findings, "duplicate_cleaning_run_reference", format!("{path}.cleaning_run_ref"),
                            "Each executed cycle must link to a unique cleaning-run identity.");
                    }
                }
                if let Some(soil_ref) = cycle.soil_application_ref.as_deref() {
                    if !soil_application_ids.insert(soil_ref) {
                        push(&mut findings, "duplicate_soil_application_reference", format!("{path}.soil_application_ref"),
                            "Each executed cycle must link to a unique soil-application identity.");
                    }
                }
                match cycle.post_clean_inspection.as_ref() {
                    Some(inspection)
                        if inspection.disposition == "pass"
                            && !is_blank(&inspection.inspector_ref)
                            && !is_blank(&inspection.evidence_ref)
                            && inspection.defect_codes.is_empty() => {}
                    Some(inspection) if inspection.disposition == "fail" => {
                        push(&mut findings, "post_clean_inspection_failed",
                            format!("{path}.post_clean_inspection"),
                            "Post-clean visual/physical inspection recorded a failure.");
                    }
                    Some(_) => push(&mut findings, "post_clean_inspection_not_admitted",
                        format!("{path}.post_clean_inspection"),
                        "Post-clean inspection must pass with inspector/evidence and no unresolved defect codes."),
                    None => push(&mut findings, "post_clean_inspection_missing",
                        format!("{path}.post_clean_inspection"),
                        "Every executed cycle requires post-clean visual/physical inspection."),
                }
            }

            let mut seen_observations = HashSet::new();
            for (obs_index, observation) in cycle.observations.iter().enumerate() {
                total_observations += 1;
                let obs_path = format!("{path}.observations[{obs_index}]");
                let Some(gate) = gate_map.get(&(observation.gate_id.as_str(), observation.zone_ref.as_str())).copied() else {
                    push(&mut findings, "observation_gate_unknown", &obs_path,
                        "Observation must link to an exact declared gate and zone.");
                    continue;
                };
                if !gate.required_sample_location_refs.contains(&observation.sample_location_ref) {
                    push(&mut findings, "observation_sample_location_unknown", format!("{obs_path}.sample_location_ref"),
                        "Observation location must be one of the gate's topology-bound required sample locations.");
                }
                if !seen_observations.insert((
                    observation.gate_id.as_str(),
                    observation.zone_ref.as_str(),
                    observation.sample_location_ref.as_str(),
                    observation.replicate_index
                )) {
                    push(&mut findings, "duplicate_observation_replicate", &obs_path,
                        "Gate/zone/sample-location/replicate identity must be unique within a cycle.");
                }
                *observation_counts.entry((
                    gate.gate_id.as_str(),
                    gate.zone_ref.as_str(),
                    observation.sample_location_ref.as_str(),
                    cycle.cycle_index
                )).or_default() += 1;
                if self.state == "proposed" {
                    push(&mut findings, "proposed_cycle_contains_measurement", &obs_path,
                        "Proposed campaign cannot include measurement observations.");
                    continue;
                }
                validate_observation(observation, gate, &obs_path, &mut findings);
            }
        }

        for gate in &self.gates {
            let mut gate_disposition = if gate.applicability == "not_applicable" {
                "not_applicable".to_string()
            } else {
                "pass".to_string()
            };
            let mut observation_count = 0usize;
            let mut passing_count = 0usize;
            let mut worst_bound: Option<f64> = None;

            if gate.applicability == "required" {
                for cycle in &self.cycles {
                    for sample_location in &gate.required_sample_location_refs {
                        let count = observation_counts.get(&(
                            gate.gate_id.as_str(),
                            gate.zone_ref.as_str(),
                            sample_location.as_str(),
                            cycle.cycle_index
                        )).copied().unwrap_or(0);
                        observation_count += count;
                        if self.state != "proposed" && count < gate.minimum_replicates_per_cycle as usize {
                            push(&mut findings, "required_gate_location_observations_missing", "cycles",
                                format!("Gate {} / zone {} / sample location {} has {count} observations at cycle {}; {} required.",
                                    gate.gate_id, gate.zone_ref, sample_location, cycle.cycle_index, gate.minimum_replicates_per_cycle));
                            gate_disposition = merge_disposition(&gate_disposition, "insufficient_evidence");
                        }
                    }
                }
                for cycle in &self.cycles {
                    for observation in &cycle.observations {
                        if observation.gate_id == gate.gate_id && observation.zone_ref == gate.zone_ref {
                            if let Some((disposition, bound)) = assess_observation(observation, gate) {
                                if disposition == "pass" { passing_count += 1; }
                                gate_disposition = merge_disposition(&gate_disposition, disposition);
                                if let Some(bound) = bound {
                                    worst_bound = match (gate.comparison_mode.as_str(), worst_bound) {
                                        ("max_residual_fraction" | "max_absolute_post", Some(existing)) => Some(existing.max(bound)),
                                        ("min_retention_fraction", Some(existing)) => Some(existing.min(bound)),
                                        (_, None) => Some(bound),
                                        _ => worst_bound,
                                    };
                                }
                            } else if self.state != "proposed" {
                                gate_disposition = merge_disposition(&gate_disposition, "insufficient_evidence");
                            }
                        }
                    }
                }
            }

            assessments.push(GateAssessment {
                gate_id: gate.gate_id.clone(),
                zone_ref: gate.zone_ref.clone(),
                disposition: gate_disposition,
                observation_count,
                passing_observation_count: passing_count,
                worst_case_bound: worst_bound,
            });
        }

        if self.state == "adjudicated" {
            match self.independent_review.as_ref() {
                Some(review) if matches!(review.disposition.as_str(), "admit" | "reject")
                    && !is_blank(&review.reviewer_ref)
                    && !is_blank(&review.reviewed_at)
                    && !is_blank(&review.review_record_ref) => {
                    if review.disposition == "admit" && assessments.iter().any(|a| {
                        a.disposition == "fail" || a.disposition == "uncertainty_blocked"
                            || a.disposition == "insufficient_evidence"
                    }) {
                        push(&mut findings, "admit_with_nonpassing_claim_gate", "independent_review",
                            "Independent review cannot admit a cleaning qualification with a nonpassing required gate.");
                    }
                }
                _ => push(&mut findings, "independent_review_missing", "independent_review",
                    "Adjudication requires an independent reviewer, timestamp, record and admit/reject disposition."),
            }
        } else if self.independent_review.is_some() {
            push(&mut findings, "premature_independent_review", "independent_review",
                "Independent review is only valid after measured execution.");
        }

        let passed_gate_count = assessments.iter().filter(|a| a.disposition == "pass").count();
        let failed_gate_count = assessments.iter().filter(|a| a.disposition == "fail").count();
        let uncertainty_blocked_gate_count = assessments.iter().filter(|a| a.disposition == "uncertainty_blocked").count();
        let insufficient_evidence_gate_count = assessments.iter().filter(|a| a.disposition == "insufficient_evidence").count();
        let not_applicable_gate_count = assessments.iter().filter(|a| a.disposition == "not_applicable").count();
        let independent_admit = self.independent_review.as_ref().is_some_and(|review| review.disposition == "admit");
        let cleaning_qualification_admitted = findings.is_empty()
            && self.state == "adjudicated"
            && independent_admit
            && assessments.iter().all(|a| matches!(a.disposition.as_str(), "pass" | "not_applicable"));

        ValidationReport {
            findings,
            cycle_count: self.cycles.len(),
            observation_count: total_observations,
            passed_gate_count,
            failed_gate_count,
            uncertainty_blocked_gate_count,
            insufficient_evidence_gate_count,
            not_applicable_gate_count,
            gate_assessments: assessments,
            cleaning_qualification_admitted,
        }
    }
}

fn validate_article(article: &ArticleBinding, state: &str, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("candidate_id", article.candidate_id.as_str()),
        ("garment_generation_ref", article.garment_generation_ref.as_str()),
        ("manufactured_article_ref", article.manufactured_article_ref.as_str()),
        ("robot_id", article.robot_id.as_str()),
        ("robot_serial", article.robot_serial.as_str()),
        ("firmware_revision", article.firmware_revision.as_str()),
        ("mission_profile_ref", article.mission_profile_ref.as_str()),
        ("asbuilt_inspection_ref", article.asbuilt_inspection_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "article_identity_missing", format!("article.{field}"),
                "Exact article, robot, mission and as-built identities are required.");
        }
    }
    for (field, digest) in [
        ("installed_article_digest", article.installed_article_digest.as_str()),
        ("robot_configuration_digest", article.robot_configuration_digest.as_str()),
        ("topology_digest", article.topology_digest.as_str()),
    ] {
        if !is_sha256(digest) {
            push(findings, "article_digest_invalid", format!("article.{field}"),
                "Article, robot-configuration and topology identity must be SHA-256 digests.");
        }
    }
    if !matches!(article.cleaning_mode.as_str(), "installed_in_place" | "removed_module") {
        push(findings, "invalid_cleaning_mode", "article.cleaning_mode",
            "Cleaning mode must be installed_in_place or removed_module.");
    }
    if article.zones.is_empty() {
        push(findings, "article_without_zones", "article.zones",
            "At least one exact functional zone is required.");
    }
    let mut zone_ids = HashSet::new();
    for (index, zone) in article.zones.iter().enumerate() {
        if is_blank(&zone.zone_ref) || !zone_ids.insert(zone.zone_ref.as_str())
            || is_blank(&zone.panel_ref) || is_blank(&zone.material_lot_ref)
            || is_blank(&zone.construction_ref)
        {
            push(findings, "zone_identity_incomplete_or_duplicate", format!("article.zones[{index}]"),
                "Each zone must uniquely bind panel, material lot and construction.");
        }
        let mut sample_locations = HashSet::new();
        if zone.cleanability_sample_location_refs.is_empty()
            || zone.cleanability_sample_location_refs.iter().any(|location| {
                is_blank(location) || !sample_locations.insert(location.as_str())
            })
        {
            push(findings, "zone_cleanability_locations_missing_or_duplicate",
                format!("article.zones[{index}].cleanability_sample_location_refs"),
                "Each zone requires unique, non-empty sample locations bound to exact topology features.");
        }
    }
    if state == "proposed" && article.installed_article_digest == "sha256:" {
        push(findings, "proposed_article_digest_missing", "article.installed_article_digest",
            "Proposed fixture must still bind the exact planned installed-article digest.");
    }
}

fn validate_application(profile: &ApplicationProfile, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("application_ref", profile.application_ref.as_str()),
        ("cleaning_requirement_ref", profile.cleaning_requirement_ref.as_str()),
        ("acceptance_owner_ref", profile.acceptance_owner_ref.as_str()),
        ("use_environment_ref", profile.use_environment_ref.as_str()),
        ("cleaning_interval_ref", profile.cleaning_interval_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "application_profile_incomplete", format!("application.{field}"),
                "Application, owner, environment, cleaning requirement and interval must be explicit.");
        }
    }
}

fn validate_soil_challenge(challenge: &SoilChallenge, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("soil_profile_ref", challenge.soil_profile_ref.as_str()),
        ("soil_class", challenge.soil_class.as_str()),
        ("soil_batch_ref", challenge.soil_batch_ref.as_str()),
        ("challenge_recipe_ref", challenge.challenge_recipe_ref.as_str()),
        ("application_method_ref", challenge.application_method_ref.as_str()),
        ("loading_unit", challenge.loading_unit.as_str()),
        ("conditioning_ref", challenge.conditioning_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "soil_challenge_identity_missing", format!("soil_challenge.{field}"),
                "Soil class, batch, recipe, application, loading unit and conditioning are required.");
        }
    }
    if !challenge.challenge_area_cm2.is_finite() || challenge.challenge_area_cm2 <= 0.0
        || !challenge.target_loading.is_finite() || challenge.target_loading <= 0.0
        || !challenge.loading_uncertainty.is_finite() || challenge.loading_uncertainty < 0.0
        || challenge.target_loading <= challenge.loading_uncertainty
        || !challenge.dwell_seconds.is_finite() || challenge.dwell_seconds < 0.0
    {
        push(findings, "soil_challenge_values_invalid", "soil_challenge",
            "Challenge area/loading/dwell must be finite and physically valid; loading must exceed its uncertainty.");
    }
    if !matches!(challenge.soil_class.as_str(), "particulate" | "oil" | "protein_surrogate" | "mineral" | "synthetic_marker" | "other_non_microbiological") {
        push(findings, "soil_class_outside_p0_scope", "soil_challenge.soil_class",
            "P0 supports declared non-microbiological soil surrogates only; microbial claims are excluded.");
    }
}

fn metric_mode_is_compatible(family: &str, mode: &str) -> bool {
    match family {
        "soil_residue" => mode == "max_residual_fraction",
        "cleaner_residue" => mode == "max_absolute_post",
        "textile_property" | "coating_integrity" | "seam_integrity" | "attachment_integrity" | "sensor_function" => mode == "min_retention_fraction",
        _ => false,
    }
}

fn validate_assay_controls(controls: &AssayControls, path: &str, findings: &mut Vec<Finding>) {
    let values_valid = [
        controls.blank_response,
        controls.blank_uncertainty,
        controls.blank_maximum,
        controls.positive_control_response,
        controls.positive_control_uncertainty,
        controls.positive_control_minimum,
        controls.recovery_fraction,
        controls.recovery_uncertainty,
        controls.recovery_minimum,
        controls.recovery_maximum,
    ].iter().all(|value| value.is_finite());
    if !values_valid || controls.blank_response < 0.0 || controls.blank_uncertainty < 0.0
        || controls.blank_maximum < 0.0 || controls.positive_control_response < 0.0
        || controls.positive_control_uncertainty < 0.0 || controls.positive_control_minimum < 0.0
        || controls.recovery_uncertainty < 0.0 || controls.recovery_minimum < 0.0
        || controls.recovery_maximum < controls.recovery_minimum
        || !is_sha256(&controls.controls_raw_data_digest)
        || is_blank(&controls.blank_control_ref) || is_blank(&controls.positive_control_ref)
        || is_blank(&controls.recovery_control_ref)
    {
        push(findings, "assay_control_record_invalid", format!("{path}.assay_controls"),
            "Assay controls require finite values, ordered recovery bounds, identity refs and a raw-data digest.");
        return;
    }
    if controls.blank_response + controls.blank_uncertainty > controls.blank_maximum {
        push(findings, "assay_blank_control_failed", format!("{path}.assay_controls.blank_response"),
            "Blank plus uncertainty exceeds the owner-defined assay blank maximum.");
    }
    if controls.positive_control_response - controls.positive_control_uncertainty < controls.positive_control_minimum {
        push(findings, "assay_positive_control_failed", format!("{path}.assay_controls.positive_control_response"),
            "Positive-control lower bound is below the owner-defined minimum.");
    }
    if controls.recovery_fraction - controls.recovery_uncertainty < controls.recovery_minimum
        || controls.recovery_fraction + controls.recovery_uncertainty > controls.recovery_maximum
    {
        push(findings, "assay_recovery_control_failed", format!("{path}.assay_controls.recovery_fraction"),
            "Recovery interval is outside owner-defined assay recovery bounds.");
    }
}

fn validate_cleaning_process(
    cycle: &CleaningCycle,
    article: &ArticleBinding,
    state: &str,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    if is_blank(&cycle.cleaning_method_selection_ref)
        || is_blank(&cycle.cleaning_method_applicability_ref)
        || is_blank(&cycle.mission_exposure_plan_ref)
        || is_blank(&cycle.cleaning_agent_ref)
        || is_blank(&cycle.cleaning_agent_batch_ref)
        || is_blank(&cycle.agent_concentration_unit)
        || is_blank(&cycle.application_mode)
        || is_blank(&cycle.rinse_water_quality_ref)
        || is_blank(&cycle.drying_method_ref)
    {
        push(findings, "cleaning_process_identity_missing", path,
            "Every cycle must bind method/applicability, mission exposure, chemical batch, concentration unit, process mode, rinse water and drying.");
    }
    if !matches!(cycle.cleaning_mode.as_str(), "installed_in_place" | "removed_module")
        || cycle.cleaning_mode != article.cleaning_mode
    {
        push(findings, "cycle_cleaning_mode_mismatch", format!("{path}.cleaning_mode"),
            "Cycle cleaning mode must match the article-level installed-in-place or removed-module mode.");
    }
    if !cycle.agent_concentration.is_finite() || cycle.agent_concentration < 0.0
        || !cycle.contact_time_seconds.is_finite() || cycle.contact_time_seconds < 0.0
        || !cycle.surface_temperature_c.is_finite()
        || !cycle.rinse_volume_l.is_finite() || cycle.rinse_volume_l < 0.0
    {
        push(findings, "cleaning_process_values_invalid", path,
            "Cleaning concentration, contact time, temperature and rinse volume must be finite and physically valid.");
    }
    if !matches!(cycle.application_mode.as_str(), "immersion" | "spray" | "wipe" | "manual_brush" | "machine_wash" | "other_declared") {
        push(findings, "unknown_cleaning_application_mode", format!("{path}.application_mode"),
            "Cleaning application mode must use the P0 declared vocabulary.");
    }
    if let Some(pressure) = cycle.application_pressure_kpa {
        if !pressure.is_finite() || pressure < 0.0 {
            push(findings, "application_pressure_invalid", format!("{path}.application_pressure_kpa"),
                "Application pressure must be finite and non-negative.");
        }
    }
    if let Some(standoff) = cycle.standoff_cm {
        if !standoff.is_finite() || standoff < 0.0 {
            push(findings, "application_standoff_invalid", format!("{path}.standoff_cm"),
                "Application standoff must be finite and non-negative.");
        }
    }
    if cycle.nozzle_ref.as_deref().is_some_and(is_blank) {
        push(findings, "blank_nozzle_reference", format!("{path}.nozzle_ref"),
            "Nozzle reference must be non-empty when supplied.");
    }
    if cycle.application_mode == "spray"
        && (cycle.application_pressure_kpa.is_none()
            || cycle.nozzle_ref.as_deref().is_none_or(is_blank)
            || cycle.standoff_cm.is_none())
    {
        push(findings, "spray_parameters_incomplete", path,
            "Spray cleaning requires measured pressure, nozzle identity and standoff.");
    }
    if state != "proposed" && cycle.robot_isolation_evidence_ref.as_deref().is_none_or(is_blank) {
        push(findings, "robot_isolation_evidence_missing", format!("{path}.robot_isolation_evidence_ref"),
            "Every executed cleaning cycle requires external robot isolation/lockout evidence.");
    }
    if article.cleaning_mode == "removed_module" {
        if state != "proposed" && (cycle.removal_record_ref.as_deref().is_none_or(is_blank)
            || cycle.reinstallation_inspection_ref.as_deref().is_none_or(is_blank))
        {
            push(findings, "removed_module_refit_evidence_missing", path,
                "Removed-module cleaning requires exact removal and reinstallation inspection evidence.");
        }
    } else if cycle.removal_record_ref.is_some() || cycle.reinstallation_inspection_ref.is_some() {
        push(findings, "unexpected_removed_module_evidence", path,
            "Installed-in-place cleaning must not carry removed-module removal/refit records.");
    }
}

fn validate_observation(
    observation: &ClaimObservation,
    gate: &ClaimGate,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    if observation.after.unit != gate.unit {
        push(findings, "after_measurement_unit_mismatch", format!("{path}.after.unit"),
            "Post-clean measurement unit must match the gate's declared unit.");
    }
    validate_measurement(&observation.after, gate, &format!("{path}.after"), findings);
    match gate.comparison_mode.as_str() {
        "max_absolute_post" => {
            if observation.before.is_some() {
                push(findings, "absolute_post_gate_has_before_measurement", path,
                    "max_absolute_post gates use the post-clean value and do not accept a ratio baseline.");
            }
        }
        "max_residual_fraction" | "min_retention_fraction" => {
            match observation.before.as_ref() {
                Some(before) => {
                    if before.unit != gate.unit {
                        push(findings, "before_measurement_unit_mismatch", format!("{path}.before.unit"),
                            "Pre-clean and post-clean measurements must use the gate's declared unit.");
                    }
                    validate_measurement(before, gate, &format!("{path}.before"), findings);
                }
                None => push(findings, "ratio_gate_before_measurement_missing", format!("{path}.before"),
                    "Ratio-based cleaning/residual/retention gates require a measured pre-clean baseline."),
            }
        }
        _ => {}
    }
}

fn validate_measurement(
    measurement: &Measurement,
    gate: &ClaimGate,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    if !measurement.value.is_finite() || measurement.value < 0.0
        || !measurement.expanded_uncertainty.is_finite() || measurement.expanded_uncertainty < 0.0
        || !measurement.detection_limit.is_finite() || measurement.detection_limit < 0.0
        || !measurement.quantification_limit.is_finite()
        || measurement.quantification_limit < measurement.detection_limit
        || is_blank(&measurement.instrument_ref) || is_blank(&measurement.calibration_ref)
        || is_blank(&measurement.method_selection_ref)
        || !is_sha256(&measurement.raw_data_digest) || is_blank(&measurement.evidence_ref)
    {
        push(findings, "measurement_record_invalid", path,
            "Measurement requires finite non-negative values, ordered LOD/LOQ, instrument/calibration/method/evidence and raw SHA-256.");
        return;
    }
    if measurement.method_selection_ref != gate.method_selection_ref {
        push(findings, "measurement_method_mismatch", format!("{path}.method_selection_ref"),
            "Measurement method must match the gate's frozen method selection.");
    }
    if measurement.calibration_state != "current" {
        push(findings, "measurement_calibration_not_current", format!("{path}.calibration_state"),
            "Expired, unknown or uncalibrated measurements cannot support a cleaning claim.");
    }
    if measurement.quantification_state != "quantified" {
        push(findings, "measurement_not_quantified", format!("{path}.quantification_state"),
            "Below-LOD, below-LOQ or above-range values are not silently promoted to zero or a pass.");
    } else if measurement.value < measurement.quantification_limit {
        push(findings, "quantified_value_below_loq", format!("{path}.value"),
            "A quantified measurement cannot be below its declared quantification limit.");
    }
}

fn assess_observation(
    observation: &ClaimObservation,
    gate: &ClaimGate,
) -> Option<(&'static str, Option<f64>)> {
    let after = &observation.after;
    if after.quantification_state != "quantified" || after.calibration_state != "current"
        || !after.value.is_finite() || !after.expanded_uncertainty.is_finite()
    {
        return Some(("insufficient_evidence", None));
    }
    match gate.comparison_mode.as_str() {
        "max_absolute_post" => {
            let point = after.value;
            let upper = after.value + after.expanded_uncertainty;
            if point > gate.acceptance_limit {
                Some(("fail", Some(upper)))
            } else if upper <= gate.acceptance_limit {
                Some(("pass", Some(upper)))
            } else {
                Some(("uncertainty_blocked", Some(upper)))
            }
        }
        "max_residual_fraction" => {
            let before = observation.before.as_ref()?;
            if before.quantification_state != "quantified" || before.calibration_state != "current"
                || before.value <= 0.0 || before.value - before.expanded_uncertainty <= 0.0
            {
                return Some(("insufficient_evidence", None));
            }
            let point = after.value / before.value;
            let upper = (after.value + after.expanded_uncertainty)
                / (before.value - before.expanded_uncertainty);
            if point > gate.acceptance_limit {
                Some(("fail", Some(upper)))
            } else if upper <= gate.acceptance_limit {
                Some(("pass", Some(upper)))
            } else {
                Some(("uncertainty_blocked", Some(upper)))
            }
        }
        "min_retention_fraction" => {
            let before = observation.before.as_ref()?;
            if before.quantification_state != "quantified" || before.calibration_state != "current"
                || before.value <= 0.0 || before.value + before.expanded_uncertainty <= 0.0
            {
                return Some(("insufficient_evidence", None));
            }
            let point = after.value / before.value;
            let lower = (after.value - after.expanded_uncertainty)
                / (before.value + before.expanded_uncertainty);
            if point < gate.acceptance_limit {
                Some(("fail", Some(lower)))
            } else if lower >= gate.acceptance_limit {
                Some(("pass", Some(lower)))
            } else {
                Some(("uncertainty_blocked", Some(lower)))
            }
        }
        _ => Some(("insufficient_evidence", None)),
    }
}

fn merge_disposition(current: &str, next: &str) -> String {
    fn rank(value: &str) -> u8 {
        match value {
            "pass" | "not_applicable" => 0,
            "uncertainty_blocked" => 1,
            "insufficient_evidence" => 2,
            "fail" => 3,
            _ => 2,
        }
    }
    if rank(next) > rank(current) { next.to_string() } else { current.to_string() }
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn push(
    findings: &mut Vec<Finding>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn digest(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn base_manifest() -> Value {
        json!({
            "schema_version": "0.1.0",
            "campaign_id": "LIGHTSHELL-CLEAN-SYNTHETIC-001",
            "state": "proposed",
            "frozen_before_first_cycle": true,
            "preregistration_ref": "SYNTHETIC-PREREGISTRATION",
            "protocol_digest": digest('a'),
            "required_cycle_count": 3,
            "target_trial_design_ref": "LIGHTSHELL-TRIAL-DESIGN-SYNTHETIC",
            "article": {
                "candidate_id": "CANDIDATE-SYNTHETIC",
                "garment_generation_ref": "GENERATION-SYNTHETIC-001",
                "manufactured_article_ref": "ARTICLE-SYNTHETIC-001",
                "installed_article_digest": digest('b'),
                "robot_id": "ROBOT-SYNTHETIC",
                "robot_serial": "ROBOT-SERIAL-SYNTHETIC",
                "firmware_revision": "FIRMWARE-SYNTHETIC",
                "robot_configuration_digest": digest('c'),
                "mission_profile_ref": "MISSION-PROFILE-SYNTHETIC",
                "topology_digest": digest('d'),
                "asbuilt_inspection_ref": "ASBUILT-INSPECTION-SYNTHETIC",
                "cleaning_mode": "installed_in_place",
                "zones": [{
                    "zone_ref": "ZONE-FOREARM-SYNTHETIC",
                    "panel_ref": "PANEL-FOREARM-SYNTHETIC",
                    "material_lot_ref": "MATERIAL-LOT-SYNTHETIC",
                    "construction_ref": "CONSTRUCTION-SYNTHETIC",
                    "cleanability_sample_location_refs": [
                        "TOPOLOGY-SURFACE-CENTER-SYNTHETIC",
                        "TOPOLOGY-SEAM-EDGE-SYNTHETIC",
                        "TOPOLOGY-FASTENER-POCKET-SYNTHETIC"
                    ]
                }]
            },
            "application": {
                "application_ref": "APPLICATION-SYNTHETIC",
                "cleaning_requirement_ref": "CLEANING-REQ-SYNTHETIC",
                "acceptance_owner_ref": "APPLICATION-OWNER-SYNTHETIC",
                "use_environment_ref": "ENVIRONMENT-SYNTHETIC",
                "cleaning_interval_ref": "INTERVAL-SYNTHETIC"
            },
            "soil_challenge": {
                "soil_profile_ref": "SOIL-PROFILE-SYNTHETIC",
                "soil_class": "synthetic_marker",
                "soil_batch_ref": "SOIL-BATCH-SYNTHETIC",
                "challenge_recipe_ref": "SOIL-RECIPE-SYNTHETIC",
                "application_method_ref": "SOIL-APPLICATION-METHOD-SYNTHETIC",
                "challenge_area_cm2": 100.0,
                "target_loading": 10.0,
                "loading_uncertainty": 0.5,
                "loading_unit": "mg",
                "dwell_seconds": 900.0,
                "conditioning_ref": "SOIL-CONDITIONING-SYNTHETIC"
            },
            "gates": [
                {
                    "gate_id": "GATE-SOIL",
                    "zone_ref": "ZONE-FOREARM-SYNTHETIC",
                    "required_sample_location_refs": [
                        "TOPOLOGY-SURFACE-CENTER-SYNTHETIC",
                        "TOPOLOGY-SEAM-EDGE-SYNTHETIC",
                        "TOPOLOGY-FASTENER-POCKET-SYNTHETIC"
                    ],
                    "metric_family": "soil_residue",
                    "comparison_mode": "max_residual_fraction",
                    "unit": "mg",
                    "acceptance_limit": 0.10,
                    "hard_gate": true,
                    "applicability": "required",
                    "applicability_owner_ref": null,
                    "not_applicable_rationale_ref": null,
                    "minimum_replicates_per_cycle": 1,
                    "method_selection_ref": "SOIL-ASSAY-METHOD-SYNTHETIC",
                    "method_applicability_ref": "SOIL-ASSAY-APPLICABILITY-SYNTHETIC",
                    "acceptance_rule_ref": "SOIL-ACCEPTANCE-RULE-SYNTHETIC",
                    "assay_controls": {
                        "blank_control_ref": "BLANK-CONTROL-SYNTHETIC",
                        "blank_response": 0.01,
                        "blank_uncertainty": 0.005,
                        "blank_maximum": 0.05,
                        "positive_control_ref": "POSITIVE-CONTROL-SYNTHETIC",
                        "positive_control_response": 9.5,
                        "positive_control_uncertainty": 0.2,
                        "positive_control_minimum": 8.0,
                        "recovery_control_ref": "RECOVERY-CONTROL-SYNTHETIC",
                        "recovery_fraction": 0.95,
                        "recovery_uncertainty": 0.02,
                        "recovery_minimum": 0.80,
                        "recovery_maximum": 1.20,
                        "controls_raw_data_digest": digest('e')
                    }
                },
                {
                    "gate_id": "GATE-SEAM",
                    "zone_ref": "ZONE-FOREARM-SYNTHETIC",
                    "required_sample_location_refs": [
                        "TOPOLOGY-SURFACE-CENTER-SYNTHETIC",
                        "TOPOLOGY-SEAM-EDGE-SYNTHETIC",
                        "TOPOLOGY-FASTENER-POCKET-SYNTHETIC"
                    ],
                    "metric_family": "seam_integrity",
                    "comparison_mode": "min_retention_fraction",
                    "unit": "N",
                    "acceptance_limit": 0.90,
                    "hard_gate": true,
                    "applicability": "required",
                    "applicability_owner_ref": null,
                    "not_applicable_rationale_ref": null,
                    "minimum_replicates_per_cycle": 1,
                    "method_selection_ref": "SEAM-TEST-METHOD-SYNTHETIC",
                    "method_applicability_ref": "SEAM-METHOD-APPLICABILITY-SYNTHETIC",
                    "acceptance_rule_ref": "SEAM-ACCEPTANCE-RULE-SYNTHETIC",
                    "assay_controls": null
                }
            ],
            "cycles": [
                {
                    "cycle_id": "CLEAN-CYCLE-0",
                    "cycle_index": 0,
                    "cleaning_method_selection_ref": "CLEAN-METHOD-SYNTHETIC",
                    "cleaning_method_applicability_ref": "CLEAN-METHOD-APP-SYNTHETIC",
                    "mission_exposure_plan_ref": "MISSION-CLEAN-EXPOSURE-SYNTHETIC",
                    "cleaning_mode": "installed_in_place",
                    "cleaning_agent_ref": "CLEANER-SYNTHETIC",
                    "cleaning_agent_batch_ref": "CLEANER-BATCH-SYNTHETIC",
                    "agent_concentration": 0.5,
                    "agent_concentration_unit": "percent",
                    "contact_time_seconds": 120.0,
                    "surface_temperature_c": 23.0,
                    "application_mode": "wipe",
                    "application_pressure_kpa": null,
                    "nozzle_ref": null,
                    "standoff_cm": null,
                    "rinse_water_quality_ref": "RINSE-WATER-QUALITY-SYNTHETIC",
                    "rinse_volume_l": 0.5,
                    "drying_method_ref": "DRYING-METHOD-SYNTHETIC",
                    "robot_isolation_evidence_ref": null,
                    "removal_record_ref": null,
                    "reinstallation_inspection_ref": null,
                    "soil_application_ref": null,
                    "cleaning_run_ref": null,
                    "process_record_digest": null,
                    "raw_process_digest": null,
                    "post_clean_inspection": null,
                    "observations": []
                },
                {
                    "cycle_id": "CLEAN-CYCLE-1",
                    "cycle_index": 1,
                    "cleaning_method_selection_ref": "CLEAN-METHOD-SYNTHETIC",
                    "cleaning_method_applicability_ref": "CLEAN-METHOD-APP-SYNTHETIC",
                    "mission_exposure_plan_ref": "MISSION-CLEAN-EXPOSURE-SYNTHETIC",
                    "cleaning_mode": "installed_in_place",
                    "cleaning_agent_ref": "CLEANER-SYNTHETIC",
                    "cleaning_agent_batch_ref": "CLEANER-BATCH-SYNTHETIC",
                    "agent_concentration": 0.5,
                    "agent_concentration_unit": "percent",
                    "contact_time_seconds": 120.0,
                    "surface_temperature_c": 23.0,
                    "application_mode": "wipe",
                    "application_pressure_kpa": null,
                    "nozzle_ref": null,
                    "standoff_cm": null,
                    "rinse_water_quality_ref": "RINSE-WATER-QUALITY-SYNTHETIC",
                    "rinse_volume_l": 0.5,
                    "drying_method_ref": "DRYING-METHOD-SYNTHETIC",
                    "robot_isolation_evidence_ref": null,
                    "removal_record_ref": null,
                    "reinstallation_inspection_ref": null,
                    "soil_application_ref": null,
                    "cleaning_run_ref": null,
                    "process_record_digest": null,
                    "raw_process_digest": null,
                    "post_clean_inspection": null,
                    "observations": []
                },
                {
                    "cycle_id": "CLEAN-CYCLE-2",
                    "cycle_index": 2,
                    "cleaning_method_selection_ref": "CLEAN-METHOD-SYNTHETIC",
                    "cleaning_method_applicability_ref": "CLEAN-METHOD-APP-SYNTHETIC",
                    "mission_exposure_plan_ref": "MISSION-CLEAN-EXPOSURE-SYNTHETIC",
                    "cleaning_mode": "installed_in_place",
                    "cleaning_agent_ref": "CLEANER-SYNTHETIC",
                    "cleaning_agent_batch_ref": "CLEANER-BATCH-SYNTHETIC",
                    "agent_concentration": 0.5,
                    "agent_concentration_unit": "percent",
                    "contact_time_seconds": 120.0,
                    "surface_temperature_c": 23.0,
                    "application_mode": "wipe",
                    "application_pressure_kpa": null,
                    "nozzle_ref": null,
                    "standoff_cm": null,
                    "rinse_water_quality_ref": "RINSE-WATER-QUALITY-SYNTHETIC",
                    "rinse_volume_l": 0.5,
                    "drying_method_ref": "DRYING-METHOD-SYNTHETIC",
                    "robot_isolation_evidence_ref": null,
                    "removal_record_ref": null,
                    "reinstallation_inspection_ref": null,
                    "soil_application_ref": null,
                    "cleaning_run_ref": null,
                    "process_record_digest": null,
                    "raw_process_digest": null,
                    "post_clean_inspection": null,
                    "observations": []
                }
            ],
            "independent_review": null
        })
    }

    fn measured_value(value: f64, uncertainty: f64, unit: &str, method: &str, character: char) -> Value {
        json!({
            "value": value,
            "expanded_uncertainty": uncertainty,
            "unit": unit,
            "quantification_state": "quantified",
            "detection_limit": 0.01,
            "quantification_limit": 0.02,
            "instrument_ref": "INSTRUMENT-SYNTHETIC",
            "calibration_ref": "CALIBRATION-SYNTHETIC",
            "calibration_state": "current",
            "method_selection_ref": method,
            "raw_data_digest": digest(character),
            "evidence_ref": format!("EVIDENCE-{}", character)
        })
    }

    fn make_measured(manifest: &mut Value) {
        manifest["state"] = json!("measured");
        for index in 0..3 {
            let cycle = &mut manifest["cycles"][index];
            cycle["robot_isolation_evidence_ref"] = json!(format!("ISOLATION-{index}"));
            cycle["soil_application_ref"] = json!(format!("SOIL-APPLICATION-{index}"));
            cycle["cleaning_run_ref"] = json!(format!("CLEANING-RUN-{index}"));
            cycle["process_record_digest"] = json!(digest(char::from(b'f' + index as u8)));
            cycle["raw_process_digest"] = json!(digest(char::from(b'i' + index as u8)));
            cycle["post_clean_inspection"] = json!({
                "disposition": "pass",
                "inspector_ref": "INSPECTOR-SYNTHETIC",
                "evidence_ref": format!("INSPECTION-{index}"),
                "defect_codes": []
            });
            let sample_locations = [
                "TOPOLOGY-SURFACE-CENTER-SYNTHETIC",
                "TOPOLOGY-SEAM-EDGE-SYNTHETIC",
                "TOPOLOGY-FASTENER-POCKET-SYNTHETIC"
            ];
            let mut observations = Vec::new();
            for (location_index, location) in sample_locations.iter().enumerate() {
                let base = b'j' + (location_index as u8 * 4);
                let soil_before = measured_value(10.0, 0.2, "mg", "SOIL-ASSAY-METHOD-SYNTHETIC", base as char);
                let soil_after = measured_value(0.5, 0.05, "mg", "SOIL-ASSAY-METHOD-SYNTHETIC", (base + 1) as char);
                let seam_before = measured_value(100.0, 1.0, "N", "SEAM-TEST-METHOD-SYNTHETIC", (base + 2) as char);
                let seam_after = measured_value(96.0, 1.0, "N", "SEAM-TEST-METHOD-SYNTHETIC", (base + 3) as char);
                observations.push(json!({
                    "gate_id": "GATE-SOIL",
                    "zone_ref": "ZONE-FOREARM-SYNTHETIC",
                    "sample_location_ref": location,
                    "replicate_index": 0,
                    "before": soil_before,
                    "after": soil_after
                }));
                observations.push(json!({
                    "gate_id": "GATE-SEAM",
                    "zone_ref": "ZONE-FOREARM-SYNTHETIC",
                    "sample_location_ref": location,
                    "replicate_index": 0,
                    "before": seam_before,
                    "after": seam_after
                }));
            }
            cycle["observations"] = Value::Array(observations);
        }
    }

    #[test]
    fn proposed_campaign_is_valid_without_physical_cleaning_claim() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.cycle_count, 3);
        assert!(!report.can_admit_cleaning_qualification());
    }

    #[test]
    fn campaign_must_include_the_pre_registered_cycle_horizon() {
        let mut manifest = base_manifest();
        manifest["required_cycle_count"] = json!(4);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "required_cycle_count_mismatch"));
    }

    #[test]
    fn executed_cycles_require_unique_run_and_soil_application_refs() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        let prior_run_ref = manifest["cycles"][1]["cleaning_run_ref"].clone();
        manifest["cycles"][2]["cleaning_run_ref"] = prior_run_ref;
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_cleaning_run_reference"));
        let mut duplicate_soil = base_manifest();
        make_measured(&mut duplicate_soil);
        let prior_soil_ref = duplicate_soil["cycles"][1]["soil_application_ref"].clone();
        duplicate_soil["cycles"][2]["soil_application_ref"] = prior_soil_ref;
        let soil_report = parse_and_validate(&duplicate_soil.to_string()).unwrap();
        assert!(soil_report.findings.iter().any(|f| f.code == "duplicate_soil_application_reference"));
    }

    #[test]
    fn measured_campaign_requires_isolation_and_process_evidence() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.observation_count, 6);
    }

    #[test]
    fn adjudicated_clean_campaign_can_be_admitted() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["state"] = json!("adjudicated");
        manifest["independent_review"] = json!({
            "disposition": "admit",
            "reviewer_ref": "INDEPENDENT-REVIEWER-SYNTHETIC",
            "reviewed_at": "2026-11-18T00:00:00Z",
            "review_record_ref": "INDEPENDENT-REVIEW-SYNTHETIC"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.can_admit_cleaning_qualification());
    }

    #[test]
    fn soil_residual_uses_conservative_upper_ratio() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["after"]["value"] = json!(0.95);
        manifest["cycles"][0]["observations"][0]["after"]["expanded_uncertainty"] = json!(0.1);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "measurement_not_quantified") == false);
        assert_eq!(report.gate_assessments.iter().find(|g| g.gate_id == "GATE-SOIL").unwrap().disposition, "uncertainty_blocked");
    }

    #[test]
    fn low_preclean_baseline_does_not_create_false_residual_pass() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["before"]["value"] = json!(0.02);
        manifest["cycles"][0]["observations"][0]["before"]["expanded_uncertainty"] = json!(0.02);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert_eq!(report.gate_assessments.iter().find(|g| g.gate_id == "GATE-SOIL").unwrap().disposition, "insufficient_evidence");
    }

    #[test]
    fn below_quantification_limit_is_not_zero_or_a_pass() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["after"]["quantification_state"] = json!("below_quantification_limit");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "measurement_not_quantified"));
        assert_eq!(report.gate_assessments.iter().find(|g| g.gate_id == "GATE-SOIL").unwrap().disposition, "insufficient_evidence");
    }

    #[test]
    fn failed_assay_blank_invalidates_soil_claim() {
        let mut manifest = base_manifest();
        manifest["gates"][0]["assay_controls"]["blank_response"] = json!(0.2);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "assay_blank_control_failed"));
    }

    #[test]
    fn failed_assay_recovery_invalidates_soil_claim() {
        let mut manifest = base_manifest();
        manifest["gates"][0]["assay_controls"]["recovery_fraction"] = json!(0.5);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "assay_recovery_control_failed"));
    }

    #[test]
    fn post_clean_agent_residue_uses_absolute_upper_bound() {
        let mut manifest = base_manifest();
        manifest["gates"].as_array_mut().unwrap().push(json!({
            "gate_id": "GATE-CLEANER-RESIDUE",
            "zone_ref": "ZONE-FOREARM-SYNTHETIC",
            "metric_family": "cleaner_residue",
            "comparison_mode": "max_absolute_post",
            "unit": "mg/m2",
            "acceptance_limit": 0.1,
            "hard_gate": true,
            "applicability": "required",
            "applicability_owner_ref": null,
            "not_applicable_rationale_ref": null,
            "minimum_replicates_per_cycle": 1,
            "method_selection_ref": "CLEANER-RESIDUE-METHOD",
            "method_applicability_ref": "CLEANER-RESIDUE-APPLICABILITY",
            "acceptance_rule_ref": "CLEANER-RESIDUE-RULE",
            "assay_controls": {
                "blank_control_ref": "BLANK",
                "blank_response": 0.001,
                "blank_uncertainty": 0.001,
                "blank_maximum": 0.01,
                "positive_control_ref": "POSITIVE",
                "positive_control_response": 1.0,
                "positive_control_uncertainty": 0.01,
                "positive_control_minimum": 0.8,
                "recovery_control_ref": "RECOVERY",
                "recovery_fraction": 0.95,
                "recovery_uncertainty": 0.02,
                "recovery_minimum": 0.8,
                "recovery_maximum": 1.2,
                "controls_raw_data_digest": digest('n')
            }
        }));
        make_measured(&mut manifest);
        for cycle in manifest["cycles"].as_array_mut().unwrap() {
            cycle["observations"].as_array_mut().unwrap().push(json!({
                "gate_id": "GATE-CLEANER-RESIDUE",
                "zone_ref": "ZONE-FOREARM-SYNTHETIC",
                "sample_location_ref": "TOPOLOGY-SURFACE-CENTER-SYNTHETIC",
                "replicate_index": 0,
                "before": null,
                "after": measured_value(0.05,0.01,"mg/m2","CLEANER-RESIDUE-METHOD",'o')
            }));
            for location in ["TOPOLOGY-SEAM-EDGE-SYNTHETIC", "TOPOLOGY-FASTENER-POCKET-SYNTHETIC"] {
                cycle["observations"].as_array_mut().unwrap().push(json!({
                    "gate_id": "GATE-CLEANER-RESIDUE",
                    "zone_ref": "ZONE-FOREARM-SYNTHETIC",
                    "sample_location_ref": location,
                    "replicate_index": 0,
                    "before": null,
                    "after": measured_value(0.05,0.01,"mg/m2","CLEANER-RESIDUE-METHOD",'o')
                }));
            }
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.gate_assessments.iter().find(|g| g.gate_id == "GATE-CLEANER-RESIDUE").unwrap().disposition, "pass");
    }

    #[test]
    fn seam_or_sensor_retention_is_not_averaged_with_cleaning() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][1]["observations"][1]["after"]["value"] = json!(70.0);
        let mut adjudicated = manifest.clone();
        adjudicated["state"] = json!("adjudicated");
        adjudicated["independent_review"] = json!({
            "disposition": "admit",
            "reviewer_ref": "REVIEWER",
            "reviewed_at": "2026-11-18T00:00:00Z",
            "review_record_ref": "REVIEW-RECEIPT"
        });
        let report = parse_and_validate(&adjudicated.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "admit_with_nonpassing_claim_gate"));
        assert!(!report.can_admit_cleaning_qualification());
    }

    #[test]
    fn hard_gate_failure_cannot_be_admitted() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["after"]["value"] = json!(5.0);
        manifest["state"] = json!("adjudicated");
        manifest["independent_review"] = json!({
            "disposition": "admit",
            "reviewer_ref": "REVIEWER",
            "reviewed_at": "2026-11-18T00:00:00Z",
            "review_record_ref": "REVIEW-RECEIPT"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "admit_with_nonpassing_claim_gate"));
        assert!(!report.can_admit_cleaning_qualification());
    }

    #[test]
    fn method_mismatch_is_rejected() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["after"]["method_selection_ref"] = json!("UNAPPROVED-METHOD");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "measurement_method_mismatch"));
    }

    #[test]
    fn expired_calibration_is_not_admitted() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["after"]["calibration_state"] = json!("expired");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "measurement_calibration_not_current"));
    }

    #[test]
    fn cleaning_mode_must_match_article_mode() {
        let mut manifest = base_manifest();
        manifest["cycles"][0]["cleaning_mode"] = json!("removed_module");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "cycle_cleaning_mode_mismatch"));
    }

    #[test]
    fn removed_module_requires_removal_and_refit_evidence() {
        let mut manifest = base_manifest();
        manifest["article"]["cleaning_mode"] = json!("removed_module");
        for cycle in manifest["cycles"].as_array_mut().unwrap() {
            cycle["cleaning_mode"] = json!("removed_module");
        }
        make_measured(&mut manifest);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "removed_module_refit_evidence_missing"));
    }

    #[test]
    fn every_cycle_requires_all_required_gate_observations() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][2]["observations"].as_array_mut().unwrap().pop();
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "required_gate_location_observations_missing"));
    }

    #[test]
    fn each_declared_cleanability_hotspot_requires_per_cycle_coverage() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"].as_array_mut().unwrap().retain(|observation| {
            observation["sample_location_ref"] != "TOPOLOGY-FASTENER-POCKET-SYNTHETIC"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "required_gate_location_observations_missing"));
    }

    #[test]
    fn observation_location_must_resolve_to_topology_sample_map() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["sample_location_ref"] = json!("UNKNOWN-POCKET");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "observation_sample_location_unknown"));
    }


        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        let first_cycle_id = manifest["cycles"][0]["cycle_id"].clone();
        manifest["cycles"][1]["cycle_id"] = first_cycle_id;
        let duplicate_observation = manifest["cycles"][1]["observations"][0].clone();
        manifest["cycles"][1]["observations"].as_array_mut().unwrap().push(duplicate_observation);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_or_blank_cycle_id"));
        assert!(report.findings.iter().any(|f| f.code == "duplicate_observation_replicate"));
    }

    #[test]
    fn unknown_zone_and_unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["gates"][0]["zone_ref"] = json!("UNKNOWN-ZONE");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "gate_zone_not_in_article"));
        manifest["unqualified_microbial_kill_claim"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }

    #[test]
    fn microbial_claims_are_outside_p0_soil_vocabulary() {
        let mut manifest = base_manifest();
        manifest["soil_challenge"]["soil_class"] = json!("microbiological");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "soil_class_outside_p0_scope"));
    }

    #[test]
    fn not_applicable_gate_requires_owner_and_reason() {
        let mut manifest = base_manifest();
        manifest["gates"][1]["applicability"] = json!("not_applicable");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "na_gate_without_owner_rationale"));
    }

    #[test]
    fn p0_required_gates_must_be_noncompensable() {
        let mut manifest = base_manifest();
        manifest["gates"][0]["hard_gate"] = json!(false);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "p0_gate_must_be_noncompensable"));
    }

    #[test]
    fn spray_cleaning_requires_pressure_nozzle_and_standoff() {
        let mut manifest = base_manifest();
        manifest["cycles"][0]["application_mode"] = json!("spray");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "spray_parameters_incomplete"));
    }

    #[test]
    fn gate_with_unknown_metric_mode_is_rejected() {
        let mut manifest = base_manifest();
        manifest["gates"][1]["metric_family"] = json!("microbial_kill");
        manifest["gates"][1]["comparison_mode"] = json!("min_retention_fraction");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unsupported_metric_comparison"));
    }

    #[test]
    fn observed_inspection_with_unknown_defect_is_not_a_pass() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["post_clean_inspection"]["defect_codes"] = json!(["RESIDUE_TRAP"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "post_clean_inspection_not_admitted"));
    }

    #[test]
    fn proposed_campaign_cannot_contain_a_cleaning_run() {
        let mut manifest = base_manifest();
        manifest["cycles"][0]["cleaning_run_ref"] = json!("PHYSICAL-RUN");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "proposed_cycle_contains_observations"));
    }

    #[test]
    fn independent_rejection_preserves_valid_non_admitted_evidence() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["cycles"][0]["observations"][0]["after"]["value"] = json!(4.0);
        manifest["state"] = json!("adjudicated");
        manifest["independent_review"] = json!({
            "disposition": "reject",
            "reviewer_ref": "REVIEWER",
            "reviewed_at": "2026-11-18T00:00:00Z",
            "review_record_ref": "REVIEW-RECEIPT"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(!report.can_admit_cleaning_qualification());
        assert_eq!(report.gate_assessments.iter().find(|g| g.gate_id == "GATE-SOIL").unwrap().disposition, "fail");
    }
}
