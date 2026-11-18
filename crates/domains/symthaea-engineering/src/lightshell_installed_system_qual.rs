// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired installed-robot / garment system qualification evidence.
//!
//! Compares a serialized robot in baseline and garment-installed conditions
//! under matched configuration, profile, pose and environment. It admits
//! claim-level non-regression evidence; it does not operate a robot, authorize
//! motion, evaluate functional safety, or certify a protective garment.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstalledSystemManifest {
    pub schema_version: String,
    pub campaign_id: String,
    pub state: String,
    pub article: ArticleBinding,
    pub test_setup: TestSetup,
    pub baseline_context: BaselineContext,
    pub gates: Vec<SystemGate>,
    pub runs: Vec<SystemRun>,
    pub engineering_review: Option<EngineeringReview>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArticleBinding {
    pub candidate_id: String,
    pub garment_generation_ref: String,
    pub manufactured_article_ref: String,
    pub robot_id: String,
    pub robot_serial: String,
    pub robot_model_revision: String,
    pub robot_firmware_revision: String,
    pub robot_configuration_digest: String,
    pub installed_garment_digest: String,
    pub topology_digest: String,
    pub route_digest: String,
    pub asbuilt_inspection_ref: String,
    pub mission_profile_ref: String,
    pub pose_set_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestSetup {
    pub test_cell_ref: String,
    pub fixture_ref: String,
    pub fixture_revision_ref: String,
    pub operating_authority_ref: String,
    pub safety_review_ref: String,
    pub approved_motion_envelope_ref: String,
    pub emergency_stop_check_ref: String,
    pub safety_systems_unchanged: bool,
    pub no_safety_bypass: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineContext {
    pub baseline_session_ref: String,
    pub robot_serial: String,
    pub firmware_revision: String,
    pub robot_configuration_digest: String,
    pub tool_payload_digest: String,
    pub mission_profile_ref: String,
    pub environment_stratum_ref: String,
    pub matched_conditions_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemGate {
    pub gate_id: String,
    pub gate_family: String,
    pub metric_id: String,
    pub unit: String,
    pub comparison_mode: String,
    pub lower_limit: Option<f64>,
    pub upper_limit: Option<f64>,
    pub hard_gate: bool,
    pub applicability: String,
    pub applicability_owner_ref: Option<String>,
    pub not_applicable_rationale_ref: Option<String>,
    pub minimum_replicates: u32,
    pub required_pose_set_ref: Option<String>,
    pub required_pose_refs: Vec<String>,
    pub method_selection_ref: String,
    pub acceptance_rule_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemRun {
    pub run_id: String,
    pub gate_id: String,
    pub paired_group_id: String,
    pub condition: String,
    pub replicate_index: u32,
    pub profile_ref: String,
    pub robot_serial: String,
    pub firmware_revision: String,
    pub robot_configuration_digest: String,
    pub tool_payload_digest: String,
    pub environment_stratum_ref: String,
    pub matched_conditions_ref: String,
    pub installed_garment_digest: Option<String>,
    pub pose_ref: Option<String>,
    pub joint_state_ref: Option<String>,
    pub measured_value: f64,
    pub expanded_uncertainty: f64,
    pub coverage_fraction: f64,
    pub evidence_origin: String,
    pub instrument_ref: String,
    pub calibration_ref: String,
    pub calibration_state: String,
    pub method_selection_ref: String,
    pub fixture_revision_ref: String,
    pub raw_data_digest: String,
    pub evidence_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineeringReview {
    pub disposition: String,
    pub reviewer_ref: String,
    pub reviewed_at: String,
    pub review_record_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub same_serial_paired_comparison_required: bool,
    pub physical_runs_require_current_calibration: bool,
    pub paired_uncertainty_uses_conservative_sum: bool,
    pub hard_gates_are_noncompensable: bool,
    pub required_pose_coverage_must_be_complete: bool,
    pub missing_or_unknown_is_not_pass: bool,
    pub no_robot_safety_authority: bool,
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
    pub disposition: String,
    pub observation_count: usize,
    pub paired_comparison_count: usize,
    pub missing_pose_refs: Vec<String>,
    pub worst_case_lower_bound: Option<f64>,
    pub worst_case_upper_bound: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub gate_count: usize,
    pub run_count: usize,
    pub passed_gate_count: usize,
    pub failed_gate_count: usize,
    pub uncertainty_blocked_gate_count: usize,
    pub insufficient_evidence_gate_count: usize,
    pub not_applicable_gate_count: usize,
    pub gate_assessments: Vec<GateAssessment>,
    pub system_acceptance_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// Acceptance is limited to the exact tested robot, garment generation,
    /// profile, environment and gate set; it is not robot safety authority.
    pub fn can_accept_installed_system(&self) -> bool {
        self.is_valid() && self.system_acceptance_admitted
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: InstalledSystemManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl InstalledSystemManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut gates: HashMap<&str, &SystemGate> = HashMap::new();
        let mut run_ids = HashSet::new();
        let mut paired_groups: HashMap<(&str, &str, Option<&str>, u32), Vec<&SystemRun>> = HashMap::new();

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell installed-system schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.campaign_id) {
            push(&mut findings, "campaign_identity_missing", "campaign_id",
                "Installed-system campaign identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "measured" | "adjudicated") {
            push(&mut findings, "invalid_campaign_state", "state",
                "Campaign state must be proposed, measured or adjudicated.");
        }
        validate_article(&self.article, &self.baseline_context, self.state.as_str(), &mut findings);
        validate_setup(&self.test_setup, self.state.as_str(), &mut findings);

        if self.gates.is_empty() {
            push(&mut findings, "campaign_without_gates", "gates",
                "Installed-system campaign requires at least one applicable or explicitly not-applicable gate.");
        }
        let mut hard_gate_count = 0usize;
        for (index, gate) in self.gates.iter().enumerate() {
            let path = format!("gates[{index}]");
            if is_blank(&gate.gate_id) || gates.insert(gate.gate_id.as_str(), gate).is_some() {
                push(&mut findings, "duplicate_or_blank_gate_id", format!("{path}.gate_id"),
                    "System gate IDs must be non-empty and unique.");
            }
            if !is_known_gate_family(&gate.gate_family) || !is_known_metric(&gate.metric_id) {
                push(&mut findings, "unknown_gate_family_or_metric", &path,
                    "Gate family and metric must use the installed-system P0 vocabulary.");
            }
            if !metric_unit_matches(&gate.metric_id, &gate.unit) {
                push(&mut findings, "gate_metric_unit_mismatch", format!("{path}.unit"),
                    "Gate unit does not match the metric's canonical unit.");
            }
            if !matches!(gate.applicability.as_str(), "required" | "not_applicable") {
                push(&mut findings, "invalid_gate_applicability", format!("{path}.applicability"),
                    "Applicability must be required or not_applicable.");
            }
            if gate.minimum_replicates == 0 {
                push(&mut findings, "zero_minimum_replicates", format!("{path}.minimum_replicates"),
                    "Applicable gates require at least one declared replicate.");
            }
            if is_blank(&gate.method_selection_ref) || is_blank(&gate.acceptance_rule_ref) {
                push(&mut findings, "gate_method_or_rule_missing", &path,
                    "Every gate must bind an exact method selection and acceptance rule.");
            }
            if !matches!(gate.comparison_mode.as_str(), "absolute_interval" | "delta_upper_bound" | "delta_lower_bound" | "delta_interval") {
                push(&mut findings, "unknown_comparison_mode", format!("{path}.comparison_mode"),
                    "Comparison mode must be absolute_interval or a paired delta bound/interval.");
            }
            if gate.applicability == "not_applicable" {
                if gate.hard_gate && (is_blank_option(&gate.applicability_owner_ref) || is_blank_option(&gate.not_applicable_rationale_ref)) {
                    push(&mut findings, "hard_gate_na_without_owner_rationale", &path,
                        "A hard gate cannot be removed without an owner and explicit rationale.");
                }
                if !gate.hard_gate && (is_blank_option(&gate.applicability_owner_ref) || is_blank_option(&gate.not_applicable_rationale_ref)) {
                    push(&mut findings, "na_without_owner_rationale", &path,
                        "A not-applicable gate requires an owner and rationale.");
                }
                if self.runs.iter().any(|run| run.gate_id == gate.gate_id) {
                    push(&mut findings, "run_attached_to_na_gate", &path,
                        "A not-applicable gate cannot have measured runs in this campaign.");
                }
            } else if gate.hard_gate {
                hard_gate_count += 1;
            }
            let paired_mode = gate.comparison_mode != "absolute_interval";
            if paired_mode && self.baseline_context.baseline_session_ref.trim().is_empty() {
                push(&mut findings, "paired_gate_without_baseline_session", &path,
                    "Paired delta gates require an exact baseline session reference.");
            }
            if matches!(gate.metric_id.as_str(), "joint_clearance_mm" | "cable_clearance_mm" | "sensor_aperture_clearance_mm")
                && gate.applicability == "required"
            {
                if is_blank_option(&gate.required_pose_set_ref) || gate.required_pose_refs.is_empty() {
                    push(&mut findings, "pose_gate_without_approved_pose_set", &path,
                        "Clearance gates require an exact approved pose-set revision and non-empty pose list.");
                }
            } else if !gate.required_pose_refs.is_empty() {
                push(&mut findings, "pose_list_on_non_pose_metric", &path,
                    "P0 pose-set coverage is reserved for clearance metrics; other time series use their profile/run identity.");
            }
            if gate.comparison_mode == "absolute_interval" || gate.comparison_mode == "delta_interval" {
                if gate.lower_limit.is_none_or(|x| !x.is_finite())
                    || gate.upper_limit.is_none_or(|x| !x.is_finite())
                    || gate.lower_limit.zip(gate.upper_limit).is_some_and(|(a, b)| a > b)
                {
                    push(&mut findings, "invalid_gate_interval", &path,
                        "Interval comparison requires finite ordered lower and upper limits.");
                }
            }
            if gate.comparison_mode == "delta_upper_bound" && gate.upper_limit.is_none_or(|x| !x.is_finite()) {
                push(&mut findings, "delta_upper_limit_missing", &path,
                    "Upper-bound comparison requires a finite maximum allowed installed-minus-baseline delta.");
            }
            if gate.comparison_mode == "delta_lower_bound" && gate.lower_limit.is_none_or(|x| !x.is_finite()) {
                push(&mut findings, "delta_lower_limit_missing", &path,
                    "Lower-bound comparison requires a finite minimum allowed installed-minus-baseline delta.");
            }
        }
        if hard_gate_count == 0 && self.state != "proposed" {
            push(&mut findings, "campaign_without_hard_gate", "gates",
                "A measured installed-system campaign must include at least one applicable hard gate.");
        }

        if self.state == "proposed" && !self.runs.is_empty() {
            push(&mut findings, "proposed_campaign_contains_runs", "runs",
                "Proposed campaign fixture cannot claim measured runs; set state to measured only with physical evidence.");
        }
        if self.state == "measured" || self.state == "adjudicated" {
            if self.runs.is_empty() {
                push(&mut findings, "measured_campaign_without_runs", "runs",
                    "Measured/adjudicated campaign requires physical system test runs.");
            }
            validate_setup(&self.test_setup, "measured", &mut findings);
        }

        for (index, run) in self.runs.iter().enumerate() {
            let path = format!("runs[{index}]");
            if is_blank(&run.run_id) || !run_ids.insert(run.run_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_run_id", format!("{path}.run_id"),
                    "System run IDs must be non-empty and unique.");
            }
            let Some(gate) = gates.get(run.gate_id.as_str()).copied() else {
                push(&mut findings, "run_unknown_gate", format!("{path}.gate_id"),
                    "System run references an undeclared gate.");
                continue;
            };
            if gate.applicability == "not_applicable" {
                continue;
            }
            if !matches!(run.condition.as_str(), "baseline_uncovered" | "garment_installed") {
                push(&mut findings, "invalid_run_condition", format!("{path}.condition"),
                    "Run condition must be baseline_uncovered or garment_installed.");
            }
            if run.replicate_index == 0 {
                push(&mut findings, "zero_replicate_index", format!("{path}.replicate_index"),
                    "Replicate indices start at one.");
            }
            if is_blank(&run.paired_group_id) || is_blank(&run.profile_ref) {
                push(&mut findings, "run_pair_or_profile_identity_missing", &path,
                    "Every run binds a paired group and exact test/mission profile.");
            }
            if run.robot_serial != self.article.robot_serial
                || run.robot_serial != self.baseline_context.robot_serial
            {
                push(&mut findings, "run_robot_serial_mismatch", format!("{path}.robot_serial"),
                    "Baseline and garment-installed runs must use the same exact robot serial as the article binding.");
            }
            if run.firmware_revision != self.article.robot_firmware_revision
                || run.firmware_revision != self.baseline_context.firmware_revision
            {
                push(&mut findings, "run_firmware_mismatch", format!("{path}.firmware_revision"),
                    "Baseline and installed runs must use the same frozen robot firmware revision.");
            }
            if run.robot_configuration_digest != self.article.robot_configuration_digest
                || run.robot_configuration_digest != self.baseline_context.robot_configuration_digest
            {
                push(&mut findings, "run_robot_configuration_mismatch", format!("{path}.robot_configuration_digest"),
                    "Baseline and installed runs must bind the same robot configuration digest.");
            }
            if run.profile_ref != self.article.mission_profile_ref
                || run.profile_ref != self.baseline_context.mission_profile_ref
            {
                push(&mut findings, "run_profile_mismatch", format!("{path}.profile_ref"),
                    "Baseline and installed runs must bind the exact same approved mission/test profile.");
            }
            if run.environment_stratum_ref != self.baseline_context.environment_stratum_ref {
                push(&mut findings, "run_environment_mismatch", format!("{path}.environment_stratum_ref"),
                    "Paired conditions must use the same declared environmental stratum.");
            }
            if run.matched_conditions_ref != self.baseline_context.matched_conditions_ref {
                push(&mut findings, "run_matched_conditions_mismatch", format!("{path}.matched_conditions_ref"),
                    "Paired runs must bind the same matched-condition record.");
            }
            if run.tool_payload_digest != self.baseline_context.tool_payload_digest {
                push(&mut findings, "run_tool_payload_mismatch", format!("{path}.tool_payload_digest"),
                    "Baseline and installed tests must use the exact same tool/payload digest.");
            }
            match run.condition.as_str() {
                "baseline_uncovered" => {
                    if run.installed_garment_digest.is_some() {
                        push(&mut findings, "baseline_run_has_garment_digest", format!("{path}.installed_garment_digest"),
                            "Uncovered baseline run cannot claim an installed garment digest.");
                    }
                }
                "garment_installed" => {
                    if run.installed_garment_digest.as_deref() != Some(self.article.installed_garment_digest.as_str()) {
                        push(&mut findings, "installed_run_garment_digest_mismatch", format!("{path}.installed_garment_digest"),
                            "Installed-condition run must bind the exact garment assembly digest.");
                    }
                }
                _ => {}
            }
            if let Some(pose_set) = gate.required_pose_set_ref.as_deref() {
                if run.pose_ref.is_none() || run.joint_state_ref.is_none() {
                    push(&mut findings, "pose_run_binding_missing", format!("{path}.pose_ref/joint_state_ref"),
                        "Clearance run requires exact pose and joint-state references.");
                }
                if run.pose_ref.as_deref().is_some_and(|pose| !gate.required_pose_refs.iter().any(|required| required == pose)) {
                    push(&mut findings, "unapproved_pose_reference", format!("{path}.pose_ref"),
                        "Run pose is not in the gate's frozen owner-approved pose set.");
                }
                if is_blank(pose_set) {
                    push(&mut findings, "blank_pose_set_ref", format!("{path}.pose_ref"),
                        "Pose-set revision must be non-empty.");
                }
            } else if run.pose_ref.is_some() || run.joint_state_ref.is_some() {
                push(&mut findings, "unexpected_pose_binding", format!("{path}.pose_ref/joint_state_ref"),
                    "Pose/joint state bindings are only used for gates with an approved pose set.");
            }
            if !run.measured_value.is_finite() || !run.expanded_uncertainty.is_finite()
                || run.expanded_uncertainty < 0.0
            {
                push(&mut findings, "invalid_run_measurement", &path,
                    "Measured value and expanded uncertainty must be finite; uncertainty must be non-negative.");
            }
            if !run.coverage_fraction.is_finite() || run.coverage_fraction <= 0.0 || run.coverage_fraction > 1.0 {
                push(&mut findings, "invalid_run_coverage", format!("{path}.coverage_fraction"),
                    "Measurement coverage must be finite and within (0, 1].");
            }
            if !matches!(run.evidence_origin.as_str(), "physical_measurement" | "synthetic_fixture" | "simulation" | "supplier_declaration") {
                push(&mut findings, "unknown_run_evidence_origin", format!("{path}.evidence_origin"),
                    "Unknown system-run evidence origin.");
            }
            if (self.state == "measured" || self.state == "adjudicated")
                && run.evidence_origin != "physical_measurement"
            {
                push(&mut findings, "nonphysical_run_promoted", format!("{path}.evidence_origin"),
                    "Measured/adjudicated installed-system runs require physical measurement evidence.");
            }
            if self.state == "measured" || self.state == "adjudicated" {
                for (field, value) in [
                    ("instrument_ref", run.instrument_ref.as_str()),
                    ("calibration_ref", run.calibration_ref.as_str()),
                    ("method_selection_ref", run.method_selection_ref.as_str()),
                    ("fixture_revision_ref", run.fixture_revision_ref.as_str()),
                    ("raw_data_digest", run.raw_data_digest.as_str()),
                    ("evidence_ref", run.evidence_ref.as_str()),
                ] {
                    if is_blank(value) {
                        push(&mut findings, "run_provenance_missing", format!("{path}.{field}"),
                            "Physical runs require instrument, calibration, method, fixture, raw data and evidence identities.");
                    }
                }
                if run.calibration_state != "valid" {
                    push(&mut findings, "run_calibration_not_valid", format!("{path}.calibration_state"),
                        "Physical installed-system runs require valid current calibration.");
                }
                if run.raw_data_digest.is_empty() || !is_sha256_digest(&run.raw_data_digest) {
                    push(&mut findings, "run_raw_digest_invalid", format!("{path}.raw_data_digest"),
                        "Raw run evidence must bind a SHA-256 digest.");
                }
                if run.method_selection_ref != gate.method_selection_ref {
                    push(&mut findings, "run_method_mismatch", format!("{path}.method_selection_ref"),
                        "Run method selection must match the gate's frozen method.");
                }
                if run.fixture_revision_ref != self.test_setup.fixture_revision_ref {
                    push(&mut findings, "run_fixture_mismatch", format!("{path}.fixture_revision_ref"),
                        "Run fixture revision must match the test setup fixture revision.");
                }
            }
            paired_groups.entry((run.gate_id.as_str(), run.paired_group_id.as_str(), run.pose_ref.as_deref(), run.replicate_index))
                .or_default().push(run);
        }

        let mut gate_assessments = Vec::new();
        for gate in &self.gates {
            if gate.applicability == "not_applicable" {
                gate_assessments.push(GateAssessment {
                    gate_id: gate.gate_id.clone(),
                    disposition: "not_applicable".to_string(),
                    observation_count: 0,
                    paired_comparison_count: 0,
                    missing_pose_refs: vec![],
                    worst_case_lower_bound: None,
                    worst_case_upper_bound: None,
                });
                continue;
            }
            let mut runs: Vec<&SystemRun> = self.runs.iter().filter(|run| run.gate_id == gate.gate_id).collect();
            let installed: Vec<&SystemRun> = runs.iter().copied().filter(|run| run.condition == "garment_installed").collect();
            let baseline: Vec<&SystemRun> = runs.iter().copied().filter(|run| run.condition == "baseline_uncovered").collect();
            let mut pair_bounds: Vec<(f64, f64)> = Vec::new();
            let mut insufficient = false;
            let mut blocked = false;
            let mut failed = false;
            let mut missing_pose_refs = Vec::new();

            if self.state == "measured" || self.state == "adjudicated" {
                if installed.len() < gate.minimum_replicates as usize {
                    insufficient = true;
                }
                if gate.comparison_mode != "absolute_interval" {
                    if baseline.len() < gate.minimum_replicates as usize {
                        insufficient = true;
                    }
                    for installed_run in &installed {
                        let match_run = baseline.iter().find(|base| {
                            base.paired_group_id == installed_run.paired_group_id
                                && base.pose_ref == installed_run.pose_ref
                                && base.replicate_index == installed_run.replicate_index
                        });
                        let Some(base) = match_run else {
                            insufficient = true;
                            continue;
                        };
                        if base.profile_ref != installed_run.profile_ref
                            || base.robot_serial != installed_run.robot_serial
                            || base.firmware_revision != installed_run.firmware_revision
                            || base.robot_configuration_digest != installed_run.robot_configuration_digest
                            || base.tool_payload_digest != installed_run.tool_payload_digest
                            || base.environment_stratum_ref != installed_run.environment_stratum_ref
                            || base.matched_conditions_ref != installed_run.matched_conditions_ref
                        {
                            push(&mut findings, "paired_run_conditions_not_matched", format!("gates.{}.paired_group_id", gate.gate_id),
                                "Baseline/installed pair differs in serial, firmware, configuration, tool/payload, profile or environment.");
                            insufficient = true;
                            continue;
                        }
                        // P0 conservative bound: do not assume independence or
                        // cancel common-mode error without a covariance receipt.
                        let delta = installed_run.measured_value - base.measured_value;
                        let combined_u = installed_run.expanded_uncertainty + base.expanded_uncertainty;
                        pair_bounds.push((delta - combined_u, delta + combined_u));
                    }
                } else {
                    for run in &installed {
                        pair_bounds.push((
                            run.measured_value - run.expanded_uncertainty,
                            run.measured_value + run.expanded_uncertainty
                        ));
                    }
                }
                for pose in &gate.required_pose_refs {
                    if !installed.iter().any(|run| run.pose_ref.as_deref() == Some(pose.as_str())) {
                        missing_pose_refs.push(pose.clone());
                    }
                }
                if !missing_pose_refs.is_empty() {
                    insufficient = true;
                    push(&mut findings, "required_pose_coverage_missing", format!("gates.{}.required_pose_refs", gate.gate_id),
                        "At least one owner-approved pose lacks an installed-condition measurement.");
                }
                if installed.iter().any(|run| run.coverage_fraction < 0.95) {
                    insufficient = true;
                }
                if gate.comparison_mode == "absolute_interval" {
                    for (lower, upper) in &pair_bounds {
                        let pass = lower >= gate.lower_limit.unwrap_or(f64::NEG_INFINITY)
                            && upper <= gate.upper_limit.unwrap_or(f64::INFINITY);
                        let fail = upper < gate.lower_limit.unwrap_or(f64::NEG_INFINITY)
                            || lower > gate.upper_limit.unwrap_or(f64::INFINITY);
                        if !pass && !fail { blocked = true; }
                        if fail { failed = true; }
                    }
                } else {
                    for (lower, upper) in &pair_bounds {
                        let pass = match gate.comparison_mode.as_str() {
                            "delta_upper_bound" => *upper <= gate.upper_limit.unwrap_or(f64::INFINITY),
                            "delta_lower_bound" => *lower >= gate.lower_limit.unwrap_or(f64::NEG_INFINITY),
                            "delta_interval" => *lower >= gate.lower_limit.unwrap_or(f64::NEG_INFINITY)
                                && *upper <= gate.upper_limit.unwrap_or(f64::INFINITY),
                            _ => false,
                        };
                        let fail = match gate.comparison_mode.as_str() {
                            "delta_upper_bound" => *lower > gate.upper_limit.unwrap_or(f64::INFINITY),
                            "delta_lower_bound" => *upper < gate.lower_limit.unwrap_or(f64::NEG_INFINITY),
                            "delta_interval" => *upper < gate.lower_limit.unwrap_or(f64::NEG_INFINITY)
                                || *lower > gate.upper_limit.unwrap_or(f64::INFINITY),
                            _ => false,
                        };
                        if !pass && !fail { blocked = true; }
                        if fail { failed = true; }
                    }
                }
            }

            let disposition = if self.state == "proposed" {
                "not_evaluated"
            } else if insufficient || pair_bounds.is_empty() {
                "insufficient_evidence"
            } else if failed {
                "fail"
            } else if blocked {
                "uncertainty_blocked"
            } else {
                "pass"
            };
            gate_assessments.push(GateAssessment {
                gate_id: gate.gate_id.clone(),
                disposition: disposition.to_string(),
                observation_count: runs.len(),
                paired_comparison_count: pair_bounds.len(),
                missing_pose_refs,
                worst_case_lower_bound: pair_bounds.iter().map(|item| item.0).reduce(f64::min),
                worst_case_upper_bound: pair_bounds.iter().map(|item| item.1).reduce(f64::max),
            });
        }

        if self.state == "adjudicated" {
            match self.engineering_review.as_ref() {
                Some(review) => {
                    if is_blank(&review.reviewer_ref) || is_blank(&review.reviewed_at) || is_blank(&review.review_record_ref) {
                        push(&mut findings, "engineering_review_provenance_missing", "engineering_review",
                            "Adjudication requires reviewer, timestamp and review-record references.");
                    }
                    if !matches!(review.disposition.as_str(), "accept" | "reject" | "hold") {
                        push(&mut findings, "invalid_engineering_review_disposition", "engineering_review.disposition",
                            "Engineering review must be accept, reject or hold.");
                    }
                    let hard_gate_nonpass = gate_assessments.iter().any(|assessment| {
                        gates.get(assessment.gate_id.as_str()).is_some_and(|gate| gate.hard_gate)
                            && !matches!(assessment.disposition.as_str(), "pass" | "not_applicable")
                    });
                    if review.disposition == "accept" && hard_gate_nonpass {
                        push(&mut findings, "accept_with_hard_gate_nonpass", "engineering_review.disposition",
                            "A campaign cannot be accepted while any applicable hard gate is failed, blocked or insufficient.");
                    }
                }
                None => push(&mut findings, "engineering_review_missing", "engineering_review",
                    "Adjudicated campaign requires an independent engineering review record."),
            }
        } else if self.engineering_review.is_some() {
            push(&mut findings, "review_before_adjudication", "engineering_review",
                "Engineering review is attached only at the adjudicated state.");
        }

        if !self.evidence_policy.same_serial_paired_comparison_required
            || !self.evidence_policy.physical_runs_require_current_calibration
            || !self.evidence_policy.paired_uncertainty_uses_conservative_sum
            || !self.evidence_policy.hard_gates_are_noncompensable
            || !self.evidence_policy.required_pose_coverage_must_be_complete
            || !self.evidence_policy.missing_or_unknown_is_not_pass
            || !self.evidence_policy.no_robot_safety_authority
        {
            push(&mut findings, "unsafe_system_evidence_policy", "evidence_policy",
                "Matched same-serial testing, conservative uncertainty, non-compensable hard gates and safety-authority separation are mandatory.");
        }

        let passed_gate_count = gate_assessments.iter().filter(|item| item.disposition == "pass").count();
        let failed_gate_count = gate_assessments.iter().filter(|item| item.disposition == "fail").count();
        let uncertainty_blocked_gate_count = gate_assessments.iter().filter(|item| item.disposition == "uncertainty_blocked").count();
        let insufficient_evidence_gate_count = gate_assessments.iter().filter(|item| item.disposition == "insufficient_evidence").count();
        let not_applicable_gate_count = gate_assessments.iter().filter(|item| item.disposition == "not_applicable").count();
        let system_acceptance_admitted = self.state == "adjudicated"
            && self.engineering_review.as_ref().is_some_and(|review| review.disposition == "accept")
            && findings.is_empty()
            && gate_assessments.iter().all(|assessment|
                matches!(assessment.disposition.as_str(), "pass" | "not_applicable"));

        ValidationReport {
            findings,
            gate_count: self.gates.len(),
            run_count: self.runs.len(),
            passed_gate_count,
            failed_gate_count,
            uncertainty_blocked_gate_count,
            insufficient_evidence_gate_count,
            not_applicable_gate_count,
            gate_assessments,
            system_acceptance_admitted,
        }
    }
}

fn validate_article(
    article: &ArticleBinding,
    baseline: &BaselineContext,
    state: &str,
    findings: &mut Vec<Finding>,
) {
    for (field, value) in [
        ("candidate_id", article.candidate_id.as_str()),
        ("garment_generation_ref", article.garment_generation_ref.as_str()),
        ("manufactured_article_ref", article.manufactured_article_ref.as_str()),
        ("robot_id", article.robot_id.as_str()),
        ("robot_serial", article.robot_serial.as_str()),
        ("robot_model_revision", article.robot_model_revision.as_str()),
        ("robot_firmware_revision", article.robot_firmware_revision.as_str()),
        ("asbuilt_inspection_ref", article.asbuilt_inspection_ref.as_str()),
        ("mission_profile_ref", article.mission_profile_ref.as_str()),
        ("pose_set_ref", article.pose_set_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "article_binding_missing", format!("article.{field}"),
                "Installed system must bind exact garment generation, robot serial/revision and upstream evidence.");
        }
    }
    for (field, value) in [
        ("robot_configuration_digest", article.robot_configuration_digest.as_str()),
        ("installed_garment_digest", article.installed_garment_digest.as_str()),
        ("topology_digest", article.topology_digest.as_str()),
        ("route_digest", article.route_digest.as_str()),
    ] {
        if !is_sha256_digest(value) {
            push(findings, "article_digest_invalid", format!("article.{field}"),
                "Robot, installed garment, topology and route identities require SHA-256 digests.");
        }
    }
    if article.robot_serial != baseline.robot_serial
        || article.robot_firmware_revision != baseline.firmware_revision
        || article.robot_configuration_digest != baseline.robot_configuration_digest
        || article.mission_profile_ref != baseline.mission_profile_ref
    {
        push(findings, "baseline_context_does_not_match_article", "baseline_context",
            "Baseline serial, firmware, robot configuration and mission profile must match the installed article campaign.");
    }
    for (field, value) in [
        ("baseline_session_ref", baseline.baseline_session_ref.as_str()),
        ("tool_payload_digest", baseline.tool_payload_digest.as_str()),
        ("environment_stratum_ref", baseline.environment_stratum_ref.as_str()),
        ("matched_conditions_ref", baseline.matched_conditions_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "baseline_binding_missing", format!("baseline_context.{field}"),
                "Baseline must bind exact session, tool/payload, environment and matched-condition records.");
        }
    }
    if state != "proposed" && baseline.baseline_session_ref.is_empty() {
        push(findings, "physical_baseline_missing", "baseline_context.baseline_session_ref",
            "Measured system qualification requires an exact uncovered baseline session.");
    }
}

fn validate_setup(setup: &TestSetup, state: &str, findings: &mut Vec<Finding>) {
    if state == "proposed" {
        return;
    }
    for (field, value) in [
        ("test_cell_ref", setup.test_cell_ref.as_str()),
        ("fixture_ref", setup.fixture_ref.as_str()),
        ("fixture_revision_ref", setup.fixture_revision_ref.as_str()),
        ("operating_authority_ref", setup.operating_authority_ref.as_str()),
        ("safety_review_ref", setup.safety_review_ref.as_str()),
        ("approved_motion_envelope_ref", setup.approved_motion_envelope_ref.as_str()),
        ("emergency_stop_check_ref", setup.emergency_stop_check_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "test_setup_provenance_missing", format!("test_setup.{field}"),
                "Measured integrated testing requires exact test-cell, fixture, operation authority, safety-review and motion-envelope references.");
        }
    }
    if !setup.safety_systems_unchanged {
        push(findings, "safety_system_configuration_changed", "test_setup.safety_systems_unchanged",
            "The test campaign cannot claim admissible system evidence when robot safety systems were modified.");
    }
    if !setup.no_safety_bypass {
        push(findings, "safety_bypass_declared", "test_setup.no_safety_bypass",
            "Any safety-system bypass invalidates the installed-system test campaign.");
    }
}

fn is_known_gate_family(value: &str) -> bool {
    matches!(value,
        "articulation_clearance" | "cable_interface" | "sensor_aperture" |
        "thermal_non_regression" | "airflow_non_regression" | "actuation_non_regression" |
        "sensing_non_regression" | "attachment_stability" | "service_access" |
        "environmental_protection" | "cleanability" | "mass_inertia")
}

fn is_known_metric(value: &str) -> bool {
    matches!(value,
        "joint_clearance_mm" | "cable_clearance_mm" | "sensor_aperture_clearance_mm" |
        "robot_temperature_delta_c" | "airflow_delta_l_min" | "pressure_drop_delta_pa" |
        "actuator_current_delta_a" | "actuator_torque_delta_nm" | "electrical_power_delta_w" |
        "sensor_signal_delta_pct" | "sensor_localization_error_mm" | "attachment_migration_mm" |
        "service_time_delta_s" | "installed_mass_delta_kg" | "inertia_delta_kg_m2" |
        "ingress_mass_gain_g" | "contamination_score_delta"
    )
}

fn metric_unit_matches(metric: &str, unit: &str) -> bool {
    let expected = match metric {
        "joint_clearance_mm" | "cable_clearance_mm" | "sensor_aperture_clearance_mm" |
        "sensor_localization_error_mm" | "attachment_migration_mm" => "mm",
        "robot_temperature_delta_c" => "degC",
        "airflow_delta_l_min" => "L/min",
        "pressure_drop_delta_pa" => "Pa",
        "actuator_current_delta_a" => "A",
        "actuator_torque_delta_nm" => "N*m",
        "electrical_power_delta_w" => "W",
        "sensor_signal_delta_pct" | "contamination_score_delta" => "%",
        "service_time_delta_s" => "s",
        "installed_mass_delta_kg" => "kg",
        "inertia_delta_kg_m2" => "kg*m^2",
        "ingress_mass_gain_g" => "g",
        _ => return false,
    };
    expected == unit
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
    value.strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn gate(id: &str, family: &str, metric: &str, unit: &str, mode: &str, lower: Option<f64>, upper: Option<f64>, pose: bool) -> Value {
        json!({
            "gate_id":id,"gate_family":family,"metric_id":metric,"unit":unit,"comparison_mode":mode,
            "lower_limit":lower,"upper_limit":upper,"hard_gate":true,"applicability":"required",
            "applicability_owner_ref":null,"not_applicable_rationale_ref":null,
            "minimum_replicates":1,
            "required_pose_set_ref":if pose {Some("POSE-SET-REV-A")} else {None},
            "required_pose_refs":if pose {vec!["POSE-EXTREME-A","POSE-EXTREME-B"]} else {vec![]},
            "method_selection_ref":format!("METHOD-{id}"),"acceptance_rule_ref":format!("RULE-{id}")
        })
    }

    fn run(id: &str, gate_id: &str, pair: &str, condition: &str, pose: Option<&str>, value: f64, uncertainty: f64) -> Value {
        let installed = condition == "garment_installed";
        json!({
            "run_id":id,"gate_id":gate_id,"paired_group_id":pair,"condition":condition,"replicate_index":1,
            "profile_ref":"MISSION-PROFILE-REV-A","robot_serial":"ROBOT-SERIAL-001",
            "firmware_revision":"FIRMWARE-REV-A","robot_configuration_digest":format!("sha256:{}", "a".repeat(64)),
            "tool_payload_digest":"TOOL-PAYLOAD-DIGEST-01","environment_stratum_ref":"ENV-STRATUM-01",
            "matched_conditions_ref":"MATCHED-CONTEXT-01",
            "installed_garment_digest":if installed {Some(format!("sha256:{}", "b".repeat(64)))} else {None},
            "pose_ref":pose,"joint_state_ref":pose.map(|p|format!("JOINT-STATE-{p}")),
            "measured_value":value,"expanded_uncertainty":uncertainty,"coverage_fraction":0.99,
            "evidence_origin":"physical_measurement","instrument_ref":"INSTRUMENT-01",
            "calibration_ref":"CALIBRATION-01","calibration_state":"valid",
            "method_selection_ref":format!("METHOD-{gate_id}"),"fixture_revision_ref":"TEST-FIXTURE-REV-A",
            "raw_data_digest":format!("sha256:{}", "c".repeat(64)),"evidence_ref":format!("EVIDENCE-{id}")
        })
    }

    fn base_manifest() -> Value {
        json!({
            "schema_version":"0.1.0","campaign_id":"LIGHTSHELL-INSTALLED-SYNTHETIC-001","state":"proposed",
            "article":{
                "candidate_id":"ELBOW-GARMENT-CANDIDATE-001","garment_generation_ref":"GARMENT-GEN-02",
                "manufactured_article_ref":"GARMENT-ARTICLE-02","robot_id":"ROBOT-001","robot_serial":"ROBOT-SERIAL-001",
                "robot_model_revision":"ROBOT-MODEL-REV-A","robot_firmware_revision":"FIRMWARE-REV-A",
                "robot_configuration_digest":format!("sha256:{}", "a".repeat(64)),
                "installed_garment_digest":format!("sha256:{}", "b".repeat(64)),
                "topology_digest":format!("sha256:{}", "d".repeat(64)),
                "route_digest":format!("sha256:{}", "e".repeat(64)),
                "asbuilt_inspection_ref":"ASBUILT-METROLOGY-12802",
                "mission_profile_ref":"MISSION-PROFILE-REV-A","pose_set_ref":"POSE-SET-REV-A"
            },
            "test_setup":{
                "test_cell_ref":"ISOLATED-TEST-CELL-01","fixture_ref":"ROBOT-FIXTURE-01",
                "fixture_revision_ref":"TEST-FIXTURE-REV-A","operating_authority_ref":"ROBOT-OWNER-AUTHORITY-01",
                "safety_review_ref":"SAFETY-REVIEW-01","approved_motion_envelope_ref":"APPROVED-MOTION-ENVELOPE-01",
                "emergency_stop_check_ref":"ESTOP-CHECK-01","safety_systems_unchanged":true,"no_safety_bypass":true
            },
            "baseline_context":{
                "baseline_session_ref":"BASELINE-SESSION-01","robot_serial":"ROBOT-SERIAL-001",
                "firmware_revision":"FIRMWARE-REV-A","robot_configuration_digest":format!("sha256:{}", "a".repeat(64)),
                "tool_payload_digest":"TOOL-PAYLOAD-DIGEST-01","mission_profile_ref":"MISSION-PROFILE-REV-A",
                "environment_stratum_ref":"ENV-STRATUM-01","matched_conditions_ref":"MATCHED-CONTEXT-01"
            },
            "gates":[
                gate("GATE-CLEARANCE","articulation_clearance","joint_clearance_mm","mm","absolute_interval",Some(2.0),Some(100.0),true),
                gate("GATE-CURRENT","actuation_non_regression","actuator_current_delta_a","A","delta_upper_bound",None,Some(0.5),false),
                gate("GATE-AIRFLOW","airflow_non_regression","airflow_delta_l_min","L/min","delta_lower_bound",Some(-10.0),None,false)
            ],
            "runs":[],
            "engineering_review":null,
            "evidence_policy":{
                "same_serial_paired_comparison_required":true,"physical_runs_require_current_calibration":true,
                "paired_uncertainty_uses_conservative_sum":true,"hard_gates_are_noncompensable":true,
                "required_pose_coverage_must_be_complete":true,"missing_or_unknown_is_not_pass":true,
                "no_robot_safety_authority":true
            }
        })
    }

    fn make_measured(manifest: &mut Value) {
        manifest["state"] = json!("measured");
        let mut runs = vec![
            run("CLEAR-A","GATE-CLEARANCE","PAIR-CLEAR-A","garment_installed",Some("POSE-EXTREME-A"),12.0,0.2),
            run("CLEAR-B","GATE-CLEARANCE","PAIR-CLEAR-B","garment_installed",Some("POSE-EXTREME-B"),9.0,0.2),
            run("CURR-BASE","GATE-CURRENT","PAIR-CURR","baseline_uncovered",None,4.0,0.05),
            run("CURR-INST","GATE-CURRENT","PAIR-CURR","garment_installed",None,4.2,0.05),
            run("AIR-BASE","GATE-AIRFLOW","PAIR-AIR","baseline_uncovered",None,100.0,1.0),
            run("AIR-INST","GATE-AIRFLOW","PAIR-AIR","garment_installed",None,96.0,1.0)
        ];
        manifest["runs"] = Value::Array(runs.drain(..).collect());
    }

    #[test]
    fn proposed_system_campaign_is_valid_without_physical_claim() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.gate_count, 3);
        assert!(!report.can_accept_installed_system());
    }

    #[test]
    fn matched_physical_system_gates_can_be_adjudicated() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["state"] = json!("adjudicated");
        manifest["engineering_review"] = json!({
            "disposition":"accept","reviewer_ref":"ENGINEER-01",
            "reviewed_at":"2026-11-18T12:00:00+02:00","review_record_ref":"SYSTEM-REVIEW-01"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.passed_gate_count, 3);
        assert!(report.can_accept_installed_system());
    }

    #[test]
    fn baseline_and_installed_runs_must_use_same_serial() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][3]["robot_serial"] = json!("ROBOT-SERIAL-OTHER");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "run_robot_serial_mismatch"));
    }

    #[test]
    fn paired_run_requires_exactly_matched_context() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][4]["environment_stratum_ref"] = json!("HOT-ENV-STRATUM");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "run_environment_mismatch"));
    }

    #[test]
    fn baseline_and_installed_run_must_share_profile_and_firmware() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][3]["profile_ref"] = json!("DIFFERENT-PROFILE");
        manifest["runs"][3]["firmware_revision"] = json!("OTHER-FIRMWARE");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "run_profile_mismatch"));
        assert!(report.findings.iter().any(|f| f.code == "run_firmware_mismatch"));
    }

    #[test]
    fn missing_approved_pose_blocks_clearance_gate() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"].as_array_mut().unwrap().retain(|run| run["pose_ref"] != "POSE-EXTREME-B");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "required_pose_coverage_missing"));
        assert_eq!(report.gate_assessments[0].disposition, "insufficient_evidence");
    }

    #[test]
    fn unapproved_pose_is_rejected() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][0]["pose_ref"] = json!("UNAPPROVED-POSE");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unapproved_pose_reference"));
    }

    #[test]
    fn conservative_paired_uncertainty_blocks_delta_near_limit() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][3]["measured_value"] = json!(4.45);
        manifest["runs"][3]["expanded_uncertainty"] = json!(0.1);
        manifest["runs"][2]["measured_value"] = json!(4.0);
        manifest["runs"][2]["expanded_uncertainty"] = json!(0.1);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        let gate = report.gate_assessments.iter().find(|g| g.gate_id == "GATE-CURRENT").unwrap();
        assert_eq!(gate.disposition, "uncertainty_blocked");
    }

    #[test]
    fn failed_hard_gate_cannot_be_averaged_away() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][3]["measured_value"] = json!(6.0);
        manifest["runs"][2]["measured_value"] = json!(4.0);
        manifest["state"] = json!("adjudicated");
        manifest["engineering_review"] = json!({
            "disposition":"accept","reviewer_ref":"ENGINEER-01",
            "reviewed_at":"2026-11-18T12:00:00+02:00","review_record_ref":"SYSTEM-REVIEW-01"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "accept_with_hard_gate_nonpass"));
        assert!(!report.can_accept_installed_system());
    }

    #[test]
    fn expired_calibration_is_not_a_physical_system_pass() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][0]["calibration_state"] = json!("expired");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "run_calibration_not_valid"));
    }

    #[test]
    fn simulation_cannot_be_promoted_to_measured_system_evidence() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][0]["evidence_origin"] = json!("simulation");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "nonphysical_run_promoted"));
    }

    #[test]
    fn safety_system_change_or_bypass_blocks_campaign() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["test_setup"]["no_safety_bypass"] = json!(false);
        manifest["test_setup"]["safety_systems_unchanged"] = json!(false);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "safety_bypass_declared"));
        assert!(report.findings.iter().any(|f| f.code == "safety_system_configuration_changed"));
    }

    #[test]
    fn not_applicable_gate_requires_owner_and_reason() {
        let mut manifest = base_manifest();
        manifest["gates"][1]["applicability"] = json!("not_applicable");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "na_without_owner_rationale"));
    }

    #[test]
    fn explicit_not_applicable_gate_is_not_silently_counted_as_pass() {
        let mut manifest = base_manifest();
        manifest["gates"][1]["applicability"] = json!("not_applicable");
        manifest["gates"][1]["applicability_owner_ref"] = json!("THERMAL-OWNER");
        manifest["gates"][1]["not_applicable_rationale_ref"] = json!("NO-ACTUATOR-UNDER-COVER");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.not_applicable_gate_count, 1);
    }

    #[test]
    fn installed_run_must_bind_exact_garment_digest() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][1]["installed_garment_digest"] = json!(format!("sha256:{}", "9".repeat(64)));
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "installed_run_garment_digest_mismatch"));
    }

    #[test]
    fn raw_data_digest_is_required() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][0]["raw_data_digest"] = json!("not-a-digest");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "run_raw_digest_invalid"));
    }

    #[test]
    fn duplicate_run_ids_are_rejected() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][1]["run_id"] = json!("CLEAR-A");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_or_blank_run_id"));
    }

    #[test]
    fn paired_gate_without_baseline_is_insufficient() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"].as_array_mut().unwrap().retain(|run| run["condition"] != "baseline_uncovered" || run["gate_id"] != "GATE-AIRFLOW");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert_eq!(report.gate_assessments.iter().find(|g| g.gate_id == "GATE-AIRFLOW").unwrap().disposition, "insufficient_evidence");
    }

    #[test]
    fn wrong_method_selection_is_rejected() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["runs"][0]["method_selection_ref"] = json!("WRONG-METHOD");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "run_method_mismatch"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["invented_robot_safety_approval"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
