// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightshell-specific robust candidate admission and interval-Pareto
//! shortlisting. Consumes frozen requirements, coupon evidence and MAT-UQ
//! receipts; it does not implement a generic optimizer or UQ engine.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustDecisionManifest {
    pub schema_version: String,
    pub decision_id: String,
    pub stage: String,
    pub platform_id: String,
    pub article_revision: Option<String>,
    pub geometry_digest: Option<String>,
    pub zone_id: String,
    pub zone_requirement_ref: Option<String>,
    pub zone_requirement_revision: Option<String>,
    pub requirements_state: String,
    pub mission_profile_ref: Option<String>,
    pub decision_domain_ref: Option<String>,
    pub confirmatory_plan_ref: Option<String>,
    pub requirements: Vec<DecisionRequirement>,
    pub candidates: Vec<Candidate>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequirement {
    pub metric_id: String,
    pub unit: String,
    pub direction: String,
    pub role: String,
    pub lower_limit: Option<f64>,
    pub upper_limit: Option<f64>,
    pub requirement_ref: Option<String>,
    pub acceptance_rule_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub candidate_id: String,
    pub construction_ref: Option<String>,
    pub material_lot_ref: Option<String>,
    pub process_route_ref: Option<String>,
    pub process_revision_ref: Option<String>,
    pub manufacturing_article_ref: Option<String>,
    pub process_capability_state: String,
    pub machine_ref: Option<String>,
    pub machine_program_revision_ref: Option<String>,
    pub process_capability_evidence_ref: Option<String>,
    pub transition_ref: Option<String>,
    pub properties: Vec<PropertyEvidence>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropertyEvidence {
    pub metric_id: String,
    pub unit: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub interval_semantics: String,
    pub coverage_fraction: Option<f64>,
    pub evidence_origin: String,
    pub applicability_state: String,
    pub uncertainty_profile_ref: Option<String>,
    pub calibration_receipt_ref: Option<String>,
    pub predictor_generation_ref: Option<String>,
    pub calibration_receipt_predictor_generation_ref: Option<String>,
    pub calibration_domain_ref: Option<String>,
    pub coupon_evidence_ref: Option<String>,
    pub coupon_run_ref: Option<String>,
    pub method_selection_ref: Option<String>,
    pub raw_data_digest: Option<String>,
    pub evidence_material_lot_ref: Option<String>,
    pub evidence_process_revision_ref: Option<String>,
    pub evidence_construction_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub unknown_is_not_zero: bool,
    pub physical_and_predicted_evidence_never_mix: bool,
    pub ood_or_unknown_applicability_blocks_prediction: bool,
    pub process_capability_required_for_physical_downselection: bool,
    pub no_weighted_aggregate_score: bool,
    pub robust_pareto_uses_interval_dominance: bool,
    pub this_is_not_robot_or_material_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateAssessment {
    pub candidate_id: String,
    pub disposition: String,
    pub blocking_metrics: Vec<String>,
    pub robust_pareto_front: bool,
    pub dominated_by: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub candidate_assessments: Vec<CandidateAssessment>,
    pub requirement_count: usize,
    pub candidate_count: usize,
    pub physical_front_count: usize,
    pub predicted_exploratory_count: usize,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// A candidate eligible for the next physical module test is not a
    /// qualified garment, PPE item, or robot safety approval.
    pub fn module_test_candidates(&self) -> Vec<&str> {
        if !self.is_valid() {
            return Vec::new();
        }
        self.candidate_assessments
            .iter()
            .filter(|candidate| candidate.disposition == "robustly_feasible_for_module_test")
            .map(|candidate| candidate.candidate_id.as_str())
            .collect()
    }

    pub fn physical_pareto_candidates(&self) -> Vec<&str> {
        if !self.is_valid() {
            return Vec::new();
        }
        self.candidate_assessments
            .iter()
            .filter(|candidate| candidate.robust_pareto_front)
            .map(|candidate| candidate.candidate_id.as_str())
            .collect()
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: RobustDecisionManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl RobustDecisionManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut requirements: HashMap<&str, &DecisionRequirement> = HashMap::new();
        let mut candidate_ids = HashSet::new();

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only robust decision schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.decision_id) || is_blank(&self.platform_id) || is_blank(&self.zone_id) {
            push(&mut findings, "missing_decision_identity", "decision_id/platform_id/zone_id",
                "Decision, platform and zone identities must be non-empty.");
        }
        if !matches!(self.stage.as_str(), "exploratory" | "downselection" | "confirmatory") {
            push(&mut findings, "invalid_decision_stage", "stage",
                "Decision stage must be exploratory, downselection or confirmatory.");
        }
        if !matches!(self.requirements_state.as_str(), "proposed" | "frozen") {
            push(&mut findings, "invalid_requirements_state", "requirements_state",
                "Requirement state must be proposed or frozen.");
        }
        if self.geometry_digest.as_deref().is_some_and(|value| !is_sha256_digest(value)) {
            push(&mut findings, "invalid_geometry_digest", "geometry_digest",
                "Geometry digest must use sha256: followed by 64 hexadecimal characters.");
        }
        if self.stage != "exploratory" && self.requirements_state != "frozen" {
            push(&mut findings, "decision_requires_frozen_requirements", "requirements_state",
                "Down-selection and confirmatory decisions require frozen zone requirements.");
        }
        if self.stage != "exploratory"
            && (is_blank_option(&self.article_revision)
                || is_blank_option(&self.geometry_digest)
                || is_blank_option(&self.zone_requirement_ref)
                || is_blank_option(&self.zone_requirement_revision)
                || is_blank_option(&self.mission_profile_ref))
        {
            push(&mut findings, "decision_missing_robot_or_requirement_binding", "article_revision/geometry_digest/zone_requirement_ref/zone_requirement_revision/mission_profile_ref",
                "A down-selection or confirmatory decision must bind the exact robot article, geometry, frozen zone requirement revision and mission profile.");
        }
        if self.stage != "exploratory" && is_blank_option(&self.decision_domain_ref) {
            push(&mut findings, "decision_domain_binding_missing", "decision_domain_ref",
                "Down-selection and confirmatory decisions must bind the target applicability domain.");
        }
        if self.stage == "confirmatory" && is_blank_option(&self.confirmatory_plan_ref) {
            push(&mut findings, "confirmatory_plan_missing", "confirmatory_plan_ref",
                "Confirmatory decisions must bind a preregistered confirmatory plan.");
        }
        if self.requirements.is_empty() {
            push(&mut findings, "no_decision_requirements", "requirements",
                "At least one zone-specific objective or hard gate is required.");
        }
        if self.candidates.is_empty() {
            push(&mut findings, "no_candidates", "candidates",
                "At least one candidate must be declared.");
        }

        let mut hard_gate_count = 0usize;
        let mut pareto_objective_count = 0usize;
        for (index, requirement) in self.requirements.iter().enumerate() {
            let path = format!("requirements[{index}]");
            if requirements.insert(requirement.metric_id.as_str(), requirement).is_some() {
                push(&mut findings, "duplicate_requirement_metric", format!("{path}.metric_id"),
                    "Each metric may appear only once in one zone decision portfolio.");
            }
            if !is_known_metric(&requirement.metric_id) {
                push(&mut findings, "unknown_decision_metric", format!("{path}.metric_id"),
                    "Metric is not in the Lightshell decision vocabulary.");
            }
            if !unit_matches_metric(&requirement.metric_id, &requirement.unit) {
                push(&mut findings, "decision_metric_unit_mismatch", format!("{path}.unit"),
                    "Metric unit does not match the canonical unit.");
            }
            if direction_for_metric(&requirement.metric_id) != Some(requirement.direction.as_str()) {
                push(&mut findings, "decision_metric_direction_mismatch", format!("{path}.direction"),
                    "Metric direction does not match its declared semantics.");
            }
            if !matches!(requirement.role.as_str(), "hard_gate" | "pareto_objective") {
                push(&mut findings, "invalid_requirement_role", format!("{path}.role"),
                    "Requirement role must be hard_gate or pareto_objective.");
            }
            if requirement.role == "hard_gate" {
                hard_gate_count += 1;
                if is_blank_option(&requirement.requirement_ref) || is_blank_option(&requirement.acceptance_rule_ref) {
                    push(&mut findings, "hard_gate_missing_authority", format!("{path}.requirement_ref/acceptance_rule_ref"),
                        "A hard gate must bind a frozen requirement and approved acceptance rule.");
                }
                if !hard_gate_bounds_are_consistent(requirement) {
                    push(&mut findings, "invalid_hard_gate_bounds", format!("{path}.lower_limit/upper_limit"),
                        "Hard-gate limits must be present and consistent with comparison direction.");
                }
            } else {
                pareto_objective_count += 1;
                if requirement.direction == "within_declared_band" {
                    push(&mut findings, "band_objective_not_pareto_orderable", format!("{path}.direction"),
                        "A bounded-band requirement belongs as a hard gate; Pareto objectives must be higher_is_better or lower_is_better.");
                }
                if requirement.lower_limit.is_some_and(|value| !value.is_finite())
                    || requirement.upper_limit.is_some_and(|value| !value.is_finite())
                    || requirement.lower_limit.zip(requirement.upper_limit).is_some_and(|(lower, upper)| lower > upper)
                {
                    push(&mut findings, "invalid_pareto_reference_bounds", format!("{path}.lower_limit/upper_limit"),
                        "Optional Pareto reference bounds must be finite and ordered when supplied.");
                }
            }
        }
        if hard_gate_count == 0 {
            push(&mut findings, "no_hard_gates", "requirements",
                "Every zone down-selection requires at least one non-compensatory hard gate.");
        }
        if pareto_objective_count == 0 {
            push(&mut findings, "no_pareto_objectives", "requirements",
                "At least one non-compensatory Pareto objective is required for candidate comparison.");
        }

        for (candidate_index, candidate) in self.candidates.iter().enumerate() {
            let path = format!("candidates[{candidate_index}]");
            if is_blank(&candidate.candidate_id) {
                push(&mut findings, "blank_candidate_id", format!("{path}.candidate_id"),
                    "Candidate identity must be non-empty.");
            }
            if !candidate_ids.insert(candidate.candidate_id.as_str()) {
                push(&mut findings, "duplicate_candidate_id", format!("{path}.candidate_id"),
                    "Candidate IDs must be unique within a zone decision.");
            }
            if is_blank_option(&candidate.construction_ref) || is_blank_option(&candidate.process_route_ref) {
                push(&mut findings, "candidate_missing_construction_or_route", format!("{path}.construction_ref/process_route_ref"),
                    "Every candidate must bind an exact construction and proposed manufacturing route.");
            }
            if !matches!(candidate.process_capability_state.as_str(), "unknown" | "pending" | "preflight_passed" | "qualified" | "not_capable") {
                push(&mut findings, "invalid_candidate_process_state", format!("{path}.process_capability_state"),
                    "Unknown manufacturing process capability state.");
            }
            if candidate.process_capability_state == "qualified" {
                for (field, value) in [
                    ("machine_ref", candidate.machine_ref.as_deref()),
                    ("machine_program_revision_ref", candidate.machine_program_revision_ref.as_deref()),
                    ("process_capability_evidence_ref", candidate.process_capability_evidence_ref.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "qualified_candidate_missing_process_identity", format!("{path}.{field}"),
                            "Qualified candidate process capability requires exact machine, program revision and capability evidence.");
                    }
                }
            }
            if candidate.properties.is_empty() {
                push(&mut findings, "candidate_without_property_evidence", format!("{path}.properties"),
                    "A candidate must carry at least one property evidence record.");
            }

            let mut evidence_metrics = HashSet::new();
            for (evidence_index, evidence) in candidate.properties.iter().enumerate() {
                let epath = format!("{path}.properties[{evidence_index}]");
                if !evidence_metrics.insert(evidence.metric_id.as_str()) {
                    push(&mut findings, "duplicate_candidate_metric_evidence", format!("{epath}.metric_id"),
                        "Candidate may have only one current aggregate evidence interval per metric in this P0 contract.");
                }
                if !requirements.contains_key(evidence.metric_id.as_str()) {
                    push(&mut findings, "candidate_evidence_not_in_requirement_portfolio", format!("{epath}.metric_id"),
                        "Candidate evidence metric is not declared by this zone decision.");
                }
                if !unit_matches_metric(&evidence.metric_id, &evidence.unit) {
                    push(&mut findings, "candidate_evidence_unit_mismatch", format!("{epath}.unit"),
                        "Candidate evidence unit does not match the canonical metric unit.");
                }
                if !evidence.lower_bound.is_finite() || !evidence.upper_bound.is_finite() || evidence.lower_bound > evidence.upper_bound {
                    push(&mut findings, "invalid_candidate_evidence_interval", format!("{epath}.lower_bound/upper_bound"),
                        "Evidence interval must be finite and ordered.");
                }
                if !matches!(evidence.interval_semantics.as_str(), "bounded_interval" | "measurement_interval" | "prediction_interval") {
                    push(&mut findings, "invalid_interval_semantics", format!("{epath}.interval_semantics"),
                        "Interval semantics must be bounded_interval, measurement_interval or prediction_interval.");
                }
                if evidence.coverage_fraction.is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0) {
                    push(&mut findings, "invalid_interval_coverage", format!("{epath}.coverage_fraction"),
                        "Coverage fraction must be finite and within (0, 1].");
                }
                if !matches!(evidence.evidence_origin.as_str(), "physical_measurement" | "calibrated_prediction" | "simulation" | "supplier_claim" | "synthetic") {
                    push(&mut findings, "invalid_evidence_origin", format!("{epath}.evidence_origin"),
                        "Unknown evidence origin.");
                }
                if !matches!(evidence.applicability_state.as_str(), "in_declared_domain" | "near_boundary" | "ood_detected" | "ood_suspected" | "unknown") {
                    push(&mut findings, "invalid_evidence_applicability", format!("{epath}.applicability_state"),
                        "Unknown evidence applicability state.");
                }

                match evidence.evidence_origin.as_str() {
                    "physical_measurement" => {
                        if evidence.interval_semantics == "prediction_interval" {
                            push(&mut findings, "physical_evidence_mislabelled_prediction_interval", format!("{epath}.interval_semantics"),
                                "Physical measurements cannot be labelled as a model prediction interval.");
                        }
                        for (field, value) in [
                            ("uncertainty_profile_ref", evidence.uncertainty_profile_ref.as_deref()),
                            ("coupon_evidence_ref", evidence.coupon_evidence_ref.as_deref()),
                            ("coupon_run_ref", evidence.coupon_run_ref.as_deref()),
                            ("method_selection_ref", evidence.method_selection_ref.as_deref()),
                            ("raw_data_digest", evidence.raw_data_digest.as_deref()),
                            ("evidence_material_lot_ref", evidence.evidence_material_lot_ref.as_deref()),
                            ("evidence_process_revision_ref", evidence.evidence_process_revision_ref.as_deref()),
                            ("evidence_construction_ref", evidence.evidence_construction_ref.as_deref()),
                        ] {
                            if value.is_none_or(is_blank) {
                                push(&mut findings, "physical_evidence_missing_provenance", format!("{epath}.{field}"),
                                    "Physical property evidence must bind coupon run, method, raw data, uncertainty and exact material/process/construction identity.");
                            }
                        }
                        if evidence.raw_data_digest.as_deref().is_some_and(|value| !is_sha256_digest(value)) {
                            push(&mut findings, "invalid_candidate_raw_digest", format!("{epath}.raw_data_digest"),
                                "Raw data digest must use sha256: followed by 64 hexadecimal characters.");
                        }
                        if evidence.applicability_state != "in_declared_domain" {
                            push(&mut findings, "physical_evidence_applicability_not_admitted", format!("{epath}.applicability_state"),
                                "Physical evidence used for robust down-selection must have an admitted method/specimen applicability state.");
                        }
                        if evidence.evidence_material_lot_ref != candidate.material_lot_ref
                            || evidence.evidence_process_revision_ref != candidate.process_revision_ref
                            || evidence.evidence_construction_ref != candidate.construction_ref
                        {
                            push(&mut findings, "candidate_evidence_identity_mismatch", epath.clone(),
                                "Physical evidence must refer to the candidate's exact material lot, process revision and construction.");
                        }
                    }
                    "calibrated_prediction" => {
                        if evidence.interval_semantics != "prediction_interval" {
                            push(&mut findings, "prediction_interval_semantics_required", format!("{epath}.interval_semantics"),
                                "Calibrated model predictions must use explicit prediction-interval semantics.");
                        }
                        if evidence.coverage_fraction.is_none() {
                            push(&mut findings, "prediction_coverage_missing", format!("{epath}.coverage_fraction"),
                                "A calibrated prediction must bind its nominal interval coverage.");
                        }
                        for (field, value) in [
                            ("uncertainty_profile_ref", evidence.uncertainty_profile_ref.as_deref()),
                            ("calibration_receipt_ref", evidence.calibration_receipt_ref.as_deref()),
                            ("predictor_generation_ref", evidence.predictor_generation_ref.as_deref()),
                            ("calibration_receipt_predictor_generation_ref", evidence.calibration_receipt_predictor_generation_ref.as_deref()),
                            ("calibration_domain_ref", evidence.calibration_domain_ref.as_deref()),
                        ] {
                            if value.is_none_or(is_blank) {
                                push(&mut findings, "prediction_missing_calibration_provenance", format!("{epath}.{field}"),
                                    "A calibrated prediction must bind the exact MAT-UQ profile, calibration receipt, predictor generation and calibration domain.");
                            }
                        }
                        if evidence.predictor_generation_ref != evidence.calibration_receipt_predictor_generation_ref {
                            push(&mut findings, "calibration_predictor_generation_mismatch", format!("{epath}.predictor_generation_ref/calibration_receipt_predictor_generation_ref"),
                                "The calibration receipt must be bound to the exact predictor generation used for this prediction.");
                        }
                        if self.decision_domain_ref.as_deref().is_some_and(|domain| evidence.calibration_domain_ref.as_deref() != Some(domain)) {
                            push(&mut findings, "prediction_calibration_domain_mismatch", format!("{epath}.calibration_domain_ref"),
                                "The calibration receipt domain must match the target decision domain.");
                        }
                    }
                    _ => {}
                }
                if evidence.applicability_state != "in_declared_domain"
                    && evidence.evidence_origin == "calibrated_prediction"
                    && self.stage != "exploratory"
                {
                    push(&mut findings, "ood_prediction_cannot_downselect", format!("{epath}.applicability_state"),
                        "Near-boundary, OOD or unknown predictions cannot support a physical down-selection.");
                }
            }
        }

        if !self.evidence_policy.unknown_is_not_zero
            || !self.evidence_policy.physical_and_predicted_evidence_never_mix
            || !self.evidence_policy.ood_or_unknown_applicability_blocks_prediction
            || !self.evidence_policy.process_capability_required_for_physical_downselection
            || !self.evidence_policy.no_weighted_aggregate_score
            || !self.evidence_policy.robust_pareto_uses_interval_dominance
            || !self.evidence_policy.this_is_not_robot_or_material_qualification
        {
            push(&mut findings, "unsafe_decision_policy", "evidence_policy",
                "Unknowns, evidence origin, OOD, process capability and interval-Pareto claim boundaries must remain mandatory.");
        }

        let mut assessments: Vec<CandidateAssessment> = self.candidates.iter()
            .map(|candidate| assess_candidate(candidate, &self.requirements, &self.stage))
            .collect();

        // Build a physical-only robust Pareto front. Prediction-only candidates
        // are never mixed into the physical front.
        let feasible_indices: Vec<usize> = assessments.iter().enumerate()
            .filter_map(|(index, assessment)| {
                (assessment.disposition == "robustly_feasible_for_module_test").then_some(index)
            })
            .collect();

        for &candidate_index in &feasible_indices {
            let mut dominated_by = Vec::new();
            for &other_index in &feasible_indices {
                if other_index == candidate_index { continue; }
                if robustly_dominates(&self.candidates[other_index], &self.candidates[candidate_index], &self.requirements) {
                    dominated_by.push(self.candidates[other_index].candidate_id.clone());
                }
            }
            dominated_by.sort();
            assessments[candidate_index].robust_pareto_front = dominated_by.is_empty();
            assessments[candidate_index].dominated_by = dominated_by;
        }

        let physical_front_count = assessments.iter().filter(|item| item.robust_pareto_front).count();
        let predicted_exploratory_count = assessments.iter()
            .filter(|item| item.disposition == "predicted_robust_feasible_exploratory").count();

        ValidationReport {
            findings,
            candidate_assessments: assessments,
            requirement_count: self.requirements.len(),
            candidate_count: self.candidates.len(),
            physical_front_count,
            predicted_exploratory_count,
        }
    }
}

fn assess_candidate(
    candidate: &Candidate,
    requirements: &[DecisionRequirement],
    stage: &str,
) -> CandidateAssessment {
    let mut assessment = CandidateAssessment {
        candidate_id: candidate.candidate_id.clone(),
        disposition: "insufficient_evidence".to_string(),
        blocking_metrics: Vec::new(),
        robust_pareto_front: false,
        dominated_by: Vec::new(),
    };
    let evidence: HashMap<&str, &PropertyEvidence> = candidate.properties.iter()
        .map(|item| (item.metric_id.as_str(), item))
        .collect();

    let mut definite_failure = false;
    let mut uncertain_gate = false;
    let mut missing = false;
    let mut applicability_blocked = false;
    let mut physical_only = true;
    let mut prediction_only = true;

    for requirement in requirements {
        let Some(item) = evidence.get(requirement.metric_id.as_str()).copied() else {
            missing = true;
            assessment.blocking_metrics.push(requirement.metric_id.clone());
            continue;
        };
        match item.evidence_origin.as_str() {
            "physical_measurement" => prediction_only = false,
            "calibrated_prediction" => physical_only = false,
            _ => {
                physical_only = false;
                prediction_only = false;
                missing = true;
                assessment.blocking_metrics.push(requirement.metric_id.clone());
                continue;
            }
        }
        if item.applicability_state != "in_declared_domain" {
            applicability_blocked = true;
            assessment.blocking_metrics.push(requirement.metric_id.clone());
        }
        if requirement.role == "hard_gate" {
            match requirement.direction.as_str() {
                "higher_is_better" => {
                    let threshold = requirement.lower_limit.unwrap_or(f64::INFINITY);
                    if item.upper_bound < threshold {
                        definite_failure = true;
                        assessment.blocking_metrics.push(requirement.metric_id.clone());
                    } else if item.lower_bound < threshold {
                        uncertain_gate = true;
                        assessment.blocking_metrics.push(requirement.metric_id.clone());
                    }
                }
                "lower_is_better" => {
                    let threshold = requirement.upper_limit.unwrap_or(f64::NEG_INFINITY);
                    if item.lower_bound > threshold {
                        definite_failure = true;
                        assessment.blocking_metrics.push(requirement.metric_id.clone());
                    } else if item.upper_bound > threshold {
                        uncertain_gate = true;
                        assessment.blocking_metrics.push(requirement.metric_id.clone());
                    }
                }
                "within_declared_band" => {
                    let lower = requirement.lower_limit.unwrap_or(f64::INFINITY);
                    let upper = requirement.upper_limit.unwrap_or(f64::NEG_INFINITY);
                    if item.upper_bound < lower || item.lower_bound > upper {
                        definite_failure = true;
                        assessment.blocking_metrics.push(requirement.metric_id.clone());
                    } else if item.lower_bound < lower || item.upper_bound > upper {
                        uncertain_gate = true;
                        assessment.blocking_metrics.push(requirement.metric_id.clone());
                    }
                }
                _ => {}
            }
        }
    }

    assessment.blocking_metrics.sort();
    assessment.blocking_metrics.dedup();

    if candidate.process_capability_state == "not_capable"
        || candidate.process_capability_state == "unknown"
        || candidate.process_capability_state == "pending"
        || (stage != "exploratory" && candidate.process_capability_state != "qualified")
    {
        assessment.disposition = "process_blocked".to_string();
        return assessment;
    }
    if definite_failure {
        assessment.disposition = "infeasible".to_string();
        return assessment;
    }
    if applicability_blocked {
        assessment.disposition = "applicability_blocked".to_string();
        return assessment;
    }
    if missing {
        assessment.disposition = "insufficient_evidence".to_string();
        return assessment;
    }
    if uncertain_gate {
        assessment.disposition = "uncertainty_blocked".to_string();
        return assessment;
    }

    if stage == "exploratory" && prediction_only {
        assessment.disposition = "predicted_robust_feasible_exploratory".to_string();
    } else if stage != "exploratory" && physical_only {
        assessment.disposition = "robustly_feasible_for_module_test".to_string();
    } else {
        assessment.disposition = "insufficient_evidence".to_string();
    }
    assessment
}

fn robustly_dominates(
    candidate_a: &Candidate,
    candidate_b: &Candidate,
    requirements: &[DecisionRequirement],
) -> bool {
    let a: HashMap<&str, &PropertyEvidence> = candidate_a.properties.iter()
        .map(|item| (item.metric_id.as_str(), item)).collect();
    let b: HashMap<&str, &PropertyEvidence> = candidate_b.properties.iter()
        .map(|item| (item.metric_id.as_str(), item)).collect();
    let mut strict_improvement = false;
    let mut compared = false;

    for requirement in requirements.iter().filter(|item| item.role == "pareto_objective") {
        let (Some(a_item), Some(b_item)) = (
            a.get(requirement.metric_id.as_str()).copied(),
            b.get(requirement.metric_id.as_str()).copied(),
        ) else { return false; };
        if a_item.evidence_origin != "physical_measurement"
            || b_item.evidence_origin != "physical_measurement"
            || a_item.applicability_state != "in_declared_domain"
            || b_item.applicability_state != "in_declared_domain"
        {
            return false;
        }
        compared = true;
        match requirement.direction.as_str() {
            // Interval A must be entirely no worse than interval B.
            "higher_is_better" => {
                if a_item.lower_bound < b_item.upper_bound { return false; }
                if a_item.lower_bound > b_item.upper_bound { strict_improvement = true; }
            }
            "lower_is_better" => {
                if a_item.upper_bound > b_item.lower_bound { return false; }
                if a_item.upper_bound < b_item.lower_bound { strict_improvement = true; }
            }
            _ => return false,
        }
    }
    compared && strict_improvement
}

fn hard_gate_bounds_are_consistent(requirement: &DecisionRequirement) -> bool {
    let lower = requirement.lower_limit;
    let upper = requirement.upper_limit;
    if lower.is_some_and(|value| !value.is_finite()) || upper.is_some_and(|value| !value.is_finite()) {
        return false;
    }
    if lower.zip(upper).is_some_and(|(lo, hi)| lo > hi) {
        return false;
    }
    match requirement.direction.as_str() {
        "higher_is_better" => lower.is_some() && upper.is_none(),
        "lower_is_better" => lower.is_none() && upper.is_some(),
        "within_declared_band" => lower.is_some() && upper.is_some(),
        _ => false,
    }
}

fn metric_definition(metric: &str) -> Option<(&'static str, &'static str)> {
    match metric {
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
        "transition_strain_concentration_ratio" => Some(("ratio", "lower_is_better")),
        "stiffness_jump_ratio" => Some(("ratio", "within_declared_band")),
        "compression_continuity_ratio" => Some(("ratio", "within_declared_band")),
        "seam_fatigue_cycles" => Some(("cycles", "higher_is_better")),
        "peel_strength_n_per_mm" => Some(("N/mm", "higher_is_better")),
        "delamination_cycles" => Some(("cycles", "higher_is_better")),
        "attachment_retention_cycles" => Some(("cycles", "higher_is_better")),
        _ => None,
    }
}

fn is_known_metric(metric: &str) -> bool {
    metric_definition(metric).is_some()
}

fn unit_matches_metric(metric: &str, unit: &str) -> bool {
    metric_definition(metric).is_some_and(|(expected, _)| expected == unit)
}

fn direction_for_metric(metric: &str) -> Option<&'static str> {
    metric_definition(metric).map(|(_, direction)| direction)
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

fn is_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn candidate(id: &str, force: [f64; 2], airflow: [f64; 2], process_state: &str) -> Value {
        let properties = [
            ("transmitted_force_n", "N", force),
            ("air_permeability_mm_s", "mm/s", airflow),
            ("added_mass_g", "g", [40.0, 44.0]),
        ];
        json!({
            "candidate_id":id,
            "construction_ref":format!("CONSTRUCTION-{id}"),
            "material_lot_ref":"LOT-001",
            "process_route_ref":format!("ROUTE-{id}"),
            "process_revision_ref":"PROCESS-REV-A",
            "manufacturing_article_ref":format!("ARTICLE-{id}"),
            "process_capability_state":process_state,
            "machine_ref":if process_state == "qualified" { Some("MACHINE-001") } else { None },
            "machine_program_revision_ref":if process_state == "qualified" { Some("PROGRAM-REV-A") } else { None },
            "process_capability_evidence_ref":if process_state == "qualified" { Some("CAPABILITY-01") } else { None },
            "transition_ref":null,
            "properties":properties.iter().map(|(metric_id,unit,bounds)| json!({
                "metric_id":metric_id,
                "unit":unit,
                "lower_bound":bounds[0],
                "upper_bound":bounds[1],
                "interval_semantics":"measurement_interval",
                "coverage_fraction":0.95,
                "evidence_origin":"physical_measurement",
                "applicability_state":"in_declared_domain",
                "uncertainty_profile_ref":"MAT-UQ-PROFILE-01",
                "calibration_receipt_ref":null,
                "predictor_generation_ref":null,
                "calibration_receipt_predictor_generation_ref":null,
                "calibration_domain_ref":null,
                "coupon_evidence_ref":"COUPON-EVIDENCE-12751",
                "coupon_run_ref":format!("RUN-{id}-{metric_id}"),
                "method_selection_ref":"METHOD-SELECTION-12759",
                "raw_data_digest":format!("sha256:{}", "a".repeat(64)),
                "evidence_material_lot_ref":"LOT-001",
                "evidence_process_revision_ref":"PROCESS-REV-A",
                "evidence_construction_ref":format!("CONSTRUCTION-{id}")
            })).collect::<Vec<_>>()
        })
    }

    fn base_manifest() -> Value {
        json!({
            "schema_version":"0.1.0",
            "decision_id":"LIGHTSHELL-DECISION-SYNTHETIC-001",
            "stage":"downselection",
            "platform_id":"SYNTHETIC-ROBOT-001",
            "article_revision":"ROBOT-REV-A",
            "geometry_digest":format!("sha256:{}", "b".repeat(64)),
            "zone_id":"forearm_shell",
            "zone_requirement_ref":"ZONE-REQ-12742",
            "zone_requirement_revision":"REV-A",
            "requirements_state":"frozen",
            "mission_profile_ref":"MISSION-PROFILE-12761",
            "decision_domain_ref":"LIGHTSHELL-DOMAIN-REV-A",
            "confirmatory_plan_ref":null,
            "requirements":[
                {
                    "metric_id":"transmitted_force_n","unit":"N","direction":"lower_is_better",
                    "role":"hard_gate","lower_limit":null,"upper_limit":80.0,
                    "requirement_ref":"ZONE-REQ-IMPACT-01","acceptance_rule_ref":"ACCEPTANCE-FORCE-01"
                },
                {
                    "metric_id":"air_permeability_mm_s","unit":"mm/s","direction":"higher_is_better",
                    "role":"pareto_objective","lower_limit":null,"upper_limit":null,
                    "requirement_ref":null,"acceptance_rule_ref":null
                },
                {
                    "metric_id":"added_mass_g","unit":"g","direction":"lower_is_better",
                    "role":"pareto_objective","lower_limit":null,"upper_limit":null,
                    "requirement_ref":null,"acceptance_rule_ref":null
                }
            ],
            "candidates":[
                candidate("A",[60.0,70.0],[120.0,130.0],"qualified"),
                candidate("B",[65.0,75.0],[100.0,110.0],"qualified"),
                candidate("C",[85.0,90.0],[140.0,150.0],"qualified")
            ],
            "evidence_policy":{
                "unknown_is_not_zero":true,
                "physical_and_predicted_evidence_never_mix":true,
                "ood_or_unknown_applicability_blocks_prediction":true,
                "process_capability_required_for_physical_downselection":true,
                "no_weighted_aggregate_score":true,
                "robust_pareto_uses_interval_dominance":true,
                "this_is_not_robot_or_material_qualification":true
            }
        })
    }


    #[test]
    fn physical_candidates_are_assessed_without_weighted_score() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.module_test_candidates(), vec!["A", "B"]);
        assert!(!report.module_test_candidates().contains(&"C"));
    }

    #[test]
    fn robust_pareto_front_preserves_incomparable_tradeoffs() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.physical_pareto_candidates().len(), 2);
        assert!(report.physical_pareto_candidates().contains(&"A"));
        assert!(report.physical_pareto_candidates().contains(&"B"));
    }

    #[test]
    fn hard_gate_interval_straddling_threshold_is_uncertainty_blocked() {
        let mut manifest = base_manifest();
        manifest["candidates"][0]["properties"][0]["lower_bound"] = json!(75.0);
        manifest["candidates"][0]["properties"][0]["upper_bound"] = json!(85.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        let a = report.candidate_assessments.iter().find(|item| item.candidate_id == "A").unwrap();
        assert_eq!(a.disposition, "uncertainty_blocked");
    }

    #[test]
    fn central_or_best_case_pass_does_not_override_violated_conservative_bound() {
        let mut manifest = base_manifest();
        manifest["candidates"][0]["properties"][0]["lower_bound"] = json!(81.0);
        manifest["candidates"][0]["properties"][0]["upper_bound"] = json!(85.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        let a = report.candidate_assessments.iter().find(|item| item.candidate_id == "A").unwrap();
        assert_eq!(a.disposition, "infeasible");
    }

    #[test]
    fn missing_hard_gate_evidence_is_insufficient_not_a_pass() {
        let mut manifest = base_manifest();
        manifest["candidates"][0]["properties"].as_array_mut().unwrap().remove(0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        let a = report.candidate_assessments.iter().find(|item| item.candidate_id == "A").unwrap();
        assert_eq!(a.disposition, "insufficient_evidence");
    }

    #[test]
    fn unqualified_manufacturing_process_blocks_physical_downselection() {
        let mut manifest = base_manifest();
        manifest["candidates"][0]["process_capability_state"] = json!("pending");
        manifest["candidates"][0]["machine_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        let a = report.candidate_assessments.iter().find(|item| item.candidate_id == "A").unwrap();
        assert_eq!(a.disposition, "process_blocked");
    }

    #[test]
    fn out_of_domain_prediction_cannot_support_downselection() {
        let mut manifest = base_manifest();
        manifest["stage"] = json!("exploratory");
        manifest["candidates"] = json!([candidate("P",[60.0,70.0],[120.0,130.0],"preflight_passed")]);
        for property in manifest["candidates"][0]["properties"].as_array_mut().unwrap() {
            property["evidence_origin"] = json!("calibrated_prediction");
            property["interval_semantics"] = json!("prediction_interval");
            property["calibration_receipt_ref"] = json!("CALIBRATION-01");
            property["predictor_generation_ref"] = json!("PREDICTOR-01");
            property["calibration_receipt_predictor_generation_ref"] = json!("PREDICTOR-01");
            property["calibration_domain_ref"] = json!("LIGHTSHELL-DOMAIN-REV-A");
            property["uncertainty_profile_ref"] = json!("MAT-UQ-01");
            property["applicability_state"] = json!("ood_detected");
            property["coupon_evidence_ref"] = json!(null);
            property["coupon_run_ref"] = json!(null);
            property["method_selection_ref"] = json!(null);
            property["raw_data_digest"] = json!(null);
            property["evidence_material_lot_ref"] = json!(null);
            property["evidence_process_revision_ref"] = json!(null);
            property["evidence_construction_ref"] = json!(null);
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        let p = &report.candidate_assessments[0];
        assert_eq!(p.disposition, "applicability_blocked");
    }

    #[test]
    fn calibrated_prediction_can_only_be_an_exploratory_shortlist() {
        let mut manifest = base_manifest();
        manifest["stage"] = json!("exploratory");
        manifest["candidates"] = json!([candidate("P",[60.0,70.0],[120.0,130.0],"preflight_passed")]);
        for property in manifest["candidates"][0]["properties"].as_array_mut().unwrap() {
            property["evidence_origin"] = json!("calibrated_prediction");
            property["interval_semantics"] = json!("prediction_interval");
            property["calibration_receipt_ref"] = json!("CALIBRATION-01");
            property["predictor_generation_ref"] = json!("PREDICTOR-01");
            property["calibration_receipt_predictor_generation_ref"] = json!("PREDICTOR-01");
            property["calibration_domain_ref"] = json!("LIGHTSHELL-DOMAIN-REV-A");
            property["uncertainty_profile_ref"] = json!("MAT-UQ-01");
            property["applicability_state"] = json!("in_declared_domain");
            property["coupon_evidence_ref"] = json!(null);
            property["coupon_run_ref"] = json!(null);
            property["method_selection_ref"] = json!(null);
            property["raw_data_digest"] = json!(null);
            property["evidence_material_lot_ref"] = json!(null);
            property["evidence_process_revision_ref"] = json!(null);
            property["evidence_construction_ref"] = json!(null);
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.candidate_assessments[0].disposition, "predicted_robust_feasible_exploratory");
        assert!(report.module_test_candidates().is_empty());
        assert_eq!(report.predicted_exploratory_count, 1);
    }

    #[test]
    fn calibration_receipt_for_another_predictor_generation_is_rejected() {
        let mut manifest = base_manifest();
        manifest["stage"] = json!("exploratory");
        manifest["candidates"] = json!([candidate("P",[60.0,70.0],[120.0,130.0],"preflight_passed")]);
        for property in manifest["candidates"][0]["properties"].as_array_mut().unwrap() {
            property["evidence_origin"] = json!("calibrated_prediction");
            property["interval_semantics"] = json!("prediction_interval");
            property["calibration_receipt_ref"] = json!("CALIBRATION-01");
            property["predictor_generation_ref"] = json!("PREDICTOR-NEW");
            property["calibration_receipt_predictor_generation_ref"] = json!("PREDICTOR-OLD");
            property["calibration_domain_ref"] = json!("LIGHTSHELL-DOMAIN-REV-A");
            property["applicability_state"] = json!("in_declared_domain");
            property["coupon_evidence_ref"] = json!(null);
            property["coupon_run_ref"] = json!(null);
            property["method_selection_ref"] = json!(null);
            property["raw_data_digest"] = json!(null);
            property["evidence_material_lot_ref"] = json!(null);
            property["evidence_process_revision_ref"] = json!(null);
            property["evidence_construction_ref"] = json!(null);
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "calibration_predictor_generation_mismatch"));
    }

    #[test]
    fn calibration_receipt_for_another_domain_is_rejected() {
        let mut manifest = base_manifest();
        manifest["stage"] = json!("exploratory");
        manifest["candidates"] = json!([candidate("P",[60.0,70.0],[120.0,130.0],"preflight_passed")]);
        for property in manifest["candidates"][0]["properties"].as_array_mut().unwrap() {
            property["evidence_origin"] = json!("calibrated_prediction");
            property["interval_semantics"] = json!("prediction_interval");
            property["calibration_receipt_ref"] = json!("CALIBRATION-01");
            property["predictor_generation_ref"] = json!("PREDICTOR-01");
            property["calibration_receipt_predictor_generation_ref"] = json!("PREDICTOR-01");
            property["calibration_domain_ref"] = json!("OTHER-DOMAIN");
            property["applicability_state"] = json!("in_declared_domain");
            property["coupon_evidence_ref"] = json!(null);
            property["coupon_run_ref"] = json!(null);
            property["method_selection_ref"] = json!(null);
            property["raw_data_digest"] = json!(null);
            property["evidence_material_lot_ref"] = json!(null);
            property["evidence_process_revision_ref"] = json!(null);
            property["evidence_construction_ref"] = json!(null);
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "prediction_calibration_domain_mismatch"));
    }

    #[test]
    fn supplier_claims_and_simulation_never_become_physical_front_members() {
        let mut manifest = base_manifest();
        manifest["candidates"][0]["properties"][0]["evidence_origin"] = json!("supplier_claim");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "physical_evidence_missing_provenance")
            || report.candidate_assessments[0].disposition == "insufficient_evidence");
        assert!(!report.physical_pareto_candidates().contains(&"A"));
    }

    #[test]
    fn physical_evidence_from_another_material_lot_is_rejected() {
        let mut manifest = base_manifest();
        manifest["candidates"][0]["properties"][0]["evidence_material_lot_ref"] = json!("LOT-OTHER");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "candidate_evidence_identity_mismatch"));
    }

    #[test]
    fn physical_evidence_from_another_process_revision_is_rejected() {
        let mut manifest = base_manifest();
        manifest["candidates"][0]["properties"][0]["evidence_process_revision_ref"] = json!("PROCESS-REV-OLD");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "candidate_evidence_identity_mismatch"));
    }

    #[test]
    fn overlapping_intervals_are_incomparable_not_arbitrarily_ranked() {
        let mut manifest = base_manifest();
        manifest["candidates"][1]["properties"][1]["lower_bound"] = json!(125.0);
        manifest["candidates"][1]["properties"][1]["upper_bound"] = json!(135.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.physical_pareto_candidates().len(), 2);
    }

    #[test]
    fn definitely_dominated_candidate_is_removed_from_front() {
        let mut manifest = base_manifest();
        manifest["candidates"][1]["properties"][1]["lower_bound"] = json!(100.0);
        manifest["candidates"][1]["properties"][1]["upper_bound"] = json!(105.0);
        manifest["candidates"][1]["properties"][2]["lower_bound"] = json!(46.0);
        manifest["candidates"][1]["properties"][2]["upper_bound"] = json!(49.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.physical_pareto_candidates().contains(&"A"));
        assert!(!report.physical_pareto_candidates().contains(&"B"));
        let b = report.candidate_assessments.iter().find(|item| item.candidate_id == "B").unwrap();
        assert_eq!(b.dominated_by, vec!["A"]);
    }

    #[test]
    fn ood_prediction_is_not_promoted_even_with_narrow_interval() {
        let mut manifest = base_manifest();
        manifest["stage"] = json!("exploratory");
        manifest["candidates"] = json!([candidate("P",[60.0,61.0],[120.0,121.0],"preflight_passed")]);
        for property in manifest["candidates"][0]["properties"].as_array_mut().unwrap() {
            property["evidence_origin"] = json!("calibrated_prediction");
            property["interval_semantics"] = json!("prediction_interval");
            property["calibration_receipt_ref"] = json!("CALIBRATION-01");
            property["predictor_generation_ref"] = json!("PREDICTOR-01");
            property["applicability_state"] = json!("ood_suspected");
            property["coupon_evidence_ref"] = json!(null);
            property["coupon_run_ref"] = json!(null);
            property["method_selection_ref"] = json!(null);
            property["raw_data_digest"] = json!(null);
            property["evidence_material_lot_ref"] = json!(null);
            property["evidence_process_revision_ref"] = json!(null);
            property["evidence_construction_ref"] = json!(null);
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert_eq!(report.candidate_assessments[0].disposition, "applicability_blocked");
    }

    #[test]
    fn downselection_requires_frozen_robot_and_zone_bindings() {
        let mut manifest = base_manifest();
        manifest["zone_requirement_revision"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "decision_missing_robot_or_requirement_binding"));
    }

    #[test]
    fn confirmatory_stage_requires_a_preregistered_plan() {
        let mut manifest = base_manifest();
        manifest["stage"] = json!("confirmatory");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "confirmatory_plan_missing"));
    }

    #[test]
    fn bounded_hard_gate_requires_a_complete_interval() {
        let mut manifest = base_manifest();
        manifest["requirements"][0]["direction"] = json!("within_declared_band");
        manifest["requirements"][0]["lower_limit"] = json!(50.0);
        manifest["requirements"][0]["upper_limit"] = json!(80.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
    }

    #[test]
    fn weighted_score_and_missing_is_zero_policies_cannot_be_disabled() {
        let mut manifest = base_manifest();
        manifest["evidence_policy"]["no_weighted_aggregate_score"] = json!(false);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unsafe_decision_policy"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["winner"] = json!("A");
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
