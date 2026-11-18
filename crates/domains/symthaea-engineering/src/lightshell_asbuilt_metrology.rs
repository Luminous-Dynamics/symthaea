// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Datum-controlled, uncertainty-guarded as-built inspection for Lightshell.
//!
//! This module admits traceable physical measurements against exact robot,
//! topology and manufacturing-route revisions. It does not acquire scans,
//! perform registration, implement a CAD kernel, grant product release or
//! certify protection/robot safety.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsBuiltInspectionManifest {
    pub schema_version: String,
    pub inspection_id: String,
    pub state: String,
    pub article: ArticleBinding,
    pub feature_catalog: Vec<FeatureSpec>,
    pub measurement_session: Option<MeasurementSession>,
    pub observations: Vec<FeatureMeasurement>,
    pub engineering_review: Option<EngineeringReview>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArticleBinding {
    pub candidate_id: String,
    pub manufactured_article_ref: String,
    pub component_revision_ref: String,
    pub route_ref: String,
    pub route_revision_ref: String,
    pub route_digest: String,
    pub topology_ref: String,
    pub topology_revision_ref: String,
    pub topology_digest: String,
    pub robot_id: String,
    pub robot_article_revision: String,
    pub geometry_digest: String,
    pub material_lot_ref: String,
    pub process_revision_ref: String,
    pub treatment_state: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSpec {
    pub feature_ref: String,
    pub feature_kind: String,
    pub nominal_geometry_ref: String,
    pub nominal_revision_ref: String,
    pub mandatory: bool,
    pub required_metrics: Vec<FeatureMetricRequirement>,
    pub required_pose_refs: Vec<String>,
    pub minimum_coverage_fraction: f64,
    pub acceptance_rule_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureMetricRequirement {
    pub metric_id: String,
    pub unit: String,
    pub lower_limit: f64,
    pub upper_limit: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementSession {
    pub session_id: String,
    pub instrument_kind: String,
    pub instrument_ref: String,
    pub calibration_ref: String,
    pub calibration_revision_ref: String,
    pub calibration_state: String,
    pub method_ref: String,
    pub protocol_ref: String,
    pub fixture_ref: String,
    pub fixture_revision_ref: String,
    pub reference_artifact_ref: String,
    pub environment_ref: String,
    pub start_time: String,
    pub end_time: String,
    pub raw_data_digest: String,
    pub parser_revision_ref: String,
    pub coordinate_frame_ref: String,
    pub registration_method: String,
    pub datum_scheme_ref: String,
    pub transform_ref: String,
    pub registration_evidence_ref: String,
    pub fit_datum_refs: Vec<String>,
    pub holdout_datum_refs: Vec<String>,
    pub holdout_datum_residuals: Vec<HoldoutDatumResidual>,
    pub excluded_roi_feature_refs: Vec<String>,
    pub maximum_holdout_residual_mm: f64,
    pub observed_holdout_residual_mm: f64,
    pub registration_uncertainty_mm: f64,
    pub uncertainty_budget_ref: String,
    pub uncertainty_budget_includes_registration: bool,
    pub coverage_fraction: f64,
    pub occlusion_fraction: f64,
    pub missing_data_fraction: f64,
    pub gap_disposition_ref: Option<String>,
    pub operator_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoldoutDatumResidual {
    pub datum_ref: String,
    pub residual_mm: f64,
    pub expanded_uncertainty_mm: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureMeasurement {
    pub measurement_id: String,
    pub feature_ref: String,
    pub metric_id: String,
    pub unit: String,
    pub pose_ref: Option<String>,
    pub joint_state_ref: Option<String>,
    pub measured_value: f64,
    pub expanded_uncertainty: f64,
    pub lower_limit: f64,
    pub upper_limit: f64,
    pub coverage_fraction: f64,
    pub measurement_origin: String,
    pub method_selection_ref: Option<String>,
    pub raw_data_slice_ref: Option<String>,
    pub uncertainty_budget_ref: Option<String>,
    pub measurement_evidence_ref: Option<String>,
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
    pub physical_measurements_require_current_calibration: bool,
    pub datum_registration_must_exclude_test_roi: bool,
    pub independent_holdout_datums_required: bool,
    pub registration_uncertainty_included_in_measurement_budget: bool,
    pub missing_coverage_never_passes: bool,
    pub guard_banded_acceptance_required: bool,
    pub engineering_adjudication_required: bool,
    pub as_built_acceptance_is_not_product_release: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureAssessment {
    pub measurement_id: String,
    pub feature_ref: String,
    pub metric_id: String,
    pub pose_ref: Option<String>,
    pub measured_lower_bound: f64,
    pub measured_upper_bound: f64,
    pub disposition: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub feature_count: usize,
    pub observation_count: usize,
    pub assessed_observation_count: usize,
    pub pass_count: usize,
    pub fail_count: usize,
    pub uncertainty_blocked_count: usize,
    pub insufficient_evidence_count: usize,
    pub feature_assessments: Vec<FeatureAssessment>,
    pub as_built_acceptance_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// Accepts only conformance to the declared nominal revision and feature
    /// scope; this is not product release, certification or robot safety.
    pub fn can_accept_as_built(&self) -> bool {
        self.is_valid() && self.as_built_acceptance_admitted
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: AsBuiltInspectionManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl AsBuiltInspectionManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut features: HashMap<&str, &FeatureSpec> = HashMap::new();
        let mut measurement_ids = HashSet::new();
        let mut feature_metric_pairs = HashSet::new();
        let mut assessments = Vec::new();

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell as-built metrology schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.inspection_id) {
            push(&mut findings, "inspection_identity_missing", "inspection_id",
                "Inspection identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "measured" | "adjudicated") {
            push(&mut findings, "invalid_inspection_state", "state",
                "Inspection state must be proposed, measured or adjudicated.");
        }
        validate_article_binding(&self.article, self.state.as_str(), &mut findings);

        if self.feature_catalog.is_empty() {
            push(&mut findings, "empty_feature_catalog", "feature_catalog",
                "As-built inspection must declare its nominal feature scope.");
        }
        if !self.feature_catalog.iter().any(|feature| feature.mandatory) {
            push(&mut findings, "no_mandatory_asbuilt_features", "feature_catalog",
                "At least one mandatory nominal feature is required for an as-built conformance decision.");
        }
        for (index, feature) in self.feature_catalog.iter().enumerate() {
            let path = format!("feature_catalog[{index}]");
            if is_blank(&feature.feature_ref) || is_blank(&feature.nominal_geometry_ref)
                || is_blank(&feature.nominal_revision_ref) || is_blank(&feature.acceptance_rule_ref)
            {
                push(&mut findings, "feature_nominal_binding_missing", &path,
                    "Feature requires exact identity, nominal geometry/revision and acceptance rule.");
            }
            if features.insert(feature.feature_ref.as_str(), feature).is_some() {
                push(&mut findings, "duplicate_feature_ref", format!("{path}.feature_ref"),
                    "Feature references must be unique.");
            }
            if !matches!(feature.feature_kind.as_str(), "panel" | "connection" | "articulation_feature" | "service_feature" | "garment_landmark") {
                push(&mut findings, "unknown_feature_kind", format!("{path}.feature_kind"),
                    "Unknown as-built feature kind.");
            }
            if feature.required_metrics.is_empty() {
                push(&mut findings, "feature_without_required_metrics", format!("{path}.required_metrics"),
                    "Each feature must declare at least one metric with frozen, feature-specific acceptance limits.");
            }
            if !feature.minimum_coverage_fraction.is_finite()
                || feature.minimum_coverage_fraction <= 0.0
                || feature.minimum_coverage_fraction > 1.0
            {
                push(&mut findings, "invalid_feature_coverage_requirement", format!("{path}.minimum_coverage_fraction"),
                    "Minimum feature coverage must be finite and within (0, 1].");
            }
            let requires_pose_coverage = feature.required_metrics.iter()
                .any(|metric| metric.metric_id == "joint_clearance_mm");
            let mut pose_refs = HashSet::new();
            for pose_ref in &feature.required_pose_refs {
                if is_blank(pose_ref) || !pose_refs.insert(pose_ref.as_str()) {
                    push(&mut findings, "duplicate_or_blank_required_pose", format!("{path}.required_pose_refs"),
                        "Required robot pose references must be non-empty and unique.");
                }
            }
            if requires_pose_coverage && feature.required_pose_refs.is_empty() {
                push(&mut findings, "joint_clearance_pose_set_missing", format!("{path}.required_pose_refs"),
                    "A joint-clearance requirement must enumerate the approved robot poses to be inspected.");
            }
            if !requires_pose_coverage && !feature.required_pose_refs.is_empty() {
                push(&mut findings, "pose_set_without_joint_clearance_metric", format!("{path}.required_pose_refs"),
                    "Pose coverage is currently supported only for joint-clearance requirements.");
            }
            let mut metrics = HashSet::new();
            for (metric_index, metric) in feature.required_metrics.iter().enumerate() {
                if !metrics.insert(metric.metric_id.as_str()) {
                    push(&mut findings, "duplicate_required_feature_metric", format!("{path}.required_metrics[{metric_index}].metric_id"),
                        "Required metric IDs must be unique within one feature.");
                }
                if !metric_unit_matches(&metric.metric_id, &metric.unit) {
                    push(&mut findings, "feature_metric_unit_mismatch", format!("{path}.required_metrics[{metric_index}].unit"),
                        "Feature requirement unit does not match the canonical metric unit.");
                }
                if !metric.lower_limit.is_finite() || !metric.upper_limit.is_finite() || metric.lower_limit > metric.upper_limit {
                    push(&mut findings, "invalid_feature_acceptance_limits", format!("{path}.required_metrics[{metric_index}]"),
                        "Feature-specific acceptance limits must be finite and ordered.");
                }
            }
        }

        if self.state == "proposed" {
            if self.measurement_session.is_some() {
                // Proposed plans may attach a session template, but must not
                // claim physical measurements as if the session were run.
                if self.observations.iter().any(|observation| observation.measurement_origin == "physical_measurement") {
                    push(&mut findings, "proposed_inspection_contains_physical_claim", "observations",
                        "A proposed inspection cannot claim physical measurements.");
                }
            }
        }

        if self.state == "measured" || self.state == "adjudicated" {
            match self.measurement_session.as_ref() {
                Some(session) => validate_session(session, &features, &self.observations, &mut findings),
                None => push(&mut findings, "measurement_session_missing", "measurement_session",
                    "Measured/adjudicated inspection requires a physical measurement session."),
            }
            if self.observations.is_empty() {
                push(&mut findings, "measured_inspection_without_observations", "observations",
                    "Measured/adjudicated inspection requires feature-level observations.");
            }
        }

        for (index, observation) in self.observations.iter().enumerate() {
            let path = format!("observations[{index}]");
            if is_blank(&observation.measurement_id) {
                push(&mut findings, "measurement_identity_missing", format!("{path}.measurement_id"),
                    "Measurement identity must be non-empty.");
            }
            if !measurement_ids.insert(observation.measurement_id.as_str()) {
                push(&mut findings, "duplicate_measurement_id", format!("{path}.measurement_id"),
                    "Measurement IDs must be unique.");
            }
            if !feature_metric_pairs.insert((
                observation.feature_ref.as_str(),
                observation.metric_id.as_str(),
                observation.pose_ref.as_deref(),
            )) {
                push(&mut findings, "duplicate_feature_metric_measurement", format!("{path}.feature_ref/metric_id/pose_ref"),
                    "A feature/metric/pose tuple may appear only once in this P0 inspection manifest.");
            }
            let Some(feature) = features.get(observation.feature_ref.as_str()).copied() else {
                push(&mut findings, "measurement_unknown_feature", format!("{path}.feature_ref"),
                    "Measurement references a feature absent from the nominal feature catalog.");
                continue;
            };
            let requirement = feature.required_metrics.iter().find(|metric| metric.metric_id == observation.metric_id);
            if requirement.is_none() {
                push(&mut findings, "measurement_metric_not_required", format!("{path}.metric_id"),
                    "Measurement metric is not declared by this feature's inspection requirements.");
            }
            if !metric_unit_matches(&observation.metric_id, &observation.unit) {
                push(&mut findings, "measurement_unit_mismatch", format!("{path}.unit"),
                    "Measurement unit does not match the canonical as-built metric unit.");
            }
            if observation.metric_id == "joint_clearance_mm" {
                if is_blank_option(&observation.pose_ref) || is_blank_option(&observation.joint_state_ref) {
                    push(&mut findings, "joint_clearance_pose_binding_missing", format!("{path}.pose_ref/joint_state_ref"),
                        "Joint-clearance measurements must bind an approved pose and exact joint-state record.");
                } else if !feature.required_pose_refs.iter().any(|pose| Some(pose.as_str()) == observation.pose_ref.as_deref()) {
                    push(&mut findings, "joint_clearance_pose_not_required", format!("{path}.pose_ref"),
                        "Joint-clearance observation pose is not in the feature's frozen required pose set.");
                }
            } else if observation.pose_ref.is_some() || observation.joint_state_ref.is_some() {
                push(&mut findings, "unexpected_pose_binding", format!("{path}.pose_ref/joint_state_ref"),
                    "Pose and joint-state references are only valid for joint-clearance measurements.");
            }
            if !observation.measured_value.is_finite()
                || !observation.expanded_uncertainty.is_finite()
                || observation.expanded_uncertainty < 0.0
                || !observation.lower_limit.is_finite()
                || !observation.upper_limit.is_finite()
                || observation.lower_limit > observation.upper_limit
            {
                push(&mut findings, "invalid_measurement_interval_inputs", &path,
                    "Measurement, uncertainty and acceptance limits must be finite and ordered; uncertainty must be non-negative.");
                continue;
            }
            if let Some(requirement) = requirement {
                if observation.unit != requirement.unit
                    || observation.lower_limit.to_bits() != requirement.lower_limit.to_bits()
                    || observation.upper_limit.to_bits() != requirement.upper_limit.to_bits()
                {
                    push(&mut findings, "measurement_acceptance_rule_mismatch", format!("{path}.lower_limit/upper_limit"),
                        "Measurement acceptance limits and unit must exactly match the feature's frozen nominal acceptance requirement.");
                }
            }
            if !observation.coverage_fraction.is_finite()
                || observation.coverage_fraction <= 0.0
                || observation.coverage_fraction > 1.0
            {
                push(&mut findings, "invalid_measurement_coverage", format!("{path}.coverage_fraction"),
                    "Measurement coverage must be finite and within (0, 1].");
            }
            if !matches!(observation.measurement_origin.as_str(), "physical_measurement" | "synthetic_fixture" | "simulation" | "supplier_claim") {
                push(&mut findings, "unknown_measurement_origin", format!("{path}.measurement_origin"),
                    "Unknown measurement evidence origin.");
            }
            if (self.state == "measured" || self.state == "adjudicated")
                && observation.measurement_origin != "physical_measurement"
            {
                push(&mut findings, "nonphysical_measurement_promoted", format!("{path}.measurement_origin"),
                    "Measured/adjudicated inspection requires physical measurement evidence.");
            }
            if self.state == "measured" || self.state == "adjudicated" {
                for (field, value) in [
                    ("method_selection_ref", observation.method_selection_ref.as_deref()),
                    ("raw_data_slice_ref", observation.raw_data_slice_ref.as_deref()),
                    ("uncertainty_budget_ref", observation.uncertainty_budget_ref.as_deref()),
                    ("measurement_evidence_ref", observation.measurement_evidence_ref.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "measurement_provenance_missing", format!("{path}.{field}"),
                            "Physical feature measurements require method, raw-data slice, uncertainty budget and evidence identity.");
                    }
                }
                if self.measurement_session.as_ref().is_some_and(|session|
                    session.uncertainty_budget_ref != observation.uncertainty_budget_ref.as_deref().unwrap_or(""))
                {
                    push(&mut findings, "measurement_uncertainty_budget_mismatch", format!("{path}.uncertainty_budget_ref"),
                        "Feature measurement must bind the exact session uncertainty budget.");
                }
                if self.measurement_session.as_ref().is_some_and(|session|
                    !session.excluded_roi_feature_refs.iter().any(|feature_ref| feature_ref == &observation.feature_ref))
                {
                    push(&mut findings, "test_roi_used_for_registration", format!("{path}.feature_ref"),
                        "The feature being inspected must be excluded from the datum-registration fit to prevent alignment laundering.");
                }
                if observation.coverage_fraction < feature.minimum_coverage_fraction {
                    push(&mut findings, "feature_coverage_below_requirement", format!("{path}.coverage_fraction"),
                        "Measured feature coverage is below the feature-specific minimum; disposition is insufficient evidence.");
                }
            }

            let lower = observation.measured_value - observation.expanded_uncertainty;
            let upper = observation.measured_value + observation.expanded_uncertainty;
            let coverage_ok = observation.coverage_fraction >= feature.minimum_coverage_fraction;
            let disposition = if !coverage_ok || observation.measurement_origin != "physical_measurement" {
                "insufficient_evidence"
            } else if lower >= observation.lower_limit && upper <= observation.upper_limit {
                "pass"
            } else if upper < observation.lower_limit || lower > observation.upper_limit {
                "fail"
            } else {
                "uncertainty_blocked"
            };
            if self.measurement_session.as_ref().is_some_and(|session|
                observation.expanded_uncertainty < session.registration_uncertainty_mm)
            {
                push(&mut findings, "feature_uncertainty_omits_registration_component", format!("{path}.expanded_uncertainty"),
                    "Feature expanded uncertainty cannot be smaller than the declared registration uncertainty contribution.");
            }
            assessments.push(FeatureAssessment {
                measurement_id: observation.measurement_id.clone(),
                feature_ref: observation.feature_ref.clone(),
                metric_id: observation.metric_id.clone(),
                pose_ref: observation.pose_ref.clone(),
                measured_lower_bound: lower,
                measured_upper_bound: upper,
                disposition: disposition.to_string(),
            });
        }

        if self.state == "measured" || self.state == "adjudicated" {
            for feature in self.feature_catalog.iter().filter(|feature| feature.mandatory) {
                for metric in &feature.required_metrics {
                    if metric.metric_id == "joint_clearance_mm" {
                        for pose_ref in &feature.required_pose_refs {
                            if !self.observations.iter().any(|observation|
                                observation.feature_ref == feature.feature_ref
                                    && observation.metric_id == metric.metric_id
                                    && observation.pose_ref.as_deref() == Some(pose_ref.as_str()))
                            {
                                push(&mut findings, "mandatory_pose_measurement_missing", "observations",
                                    format!("Mandatory feature {} is missing joint-clearance measurement at pose {}.", feature.feature_ref, pose_ref));
                            }
                        }
                    } else if !self.observations.iter().any(|observation|
                        observation.feature_ref == feature.feature_ref && observation.metric_id == metric.metric_id)
                    {
                        push(&mut findings, "mandatory_feature_measurement_missing", "observations",
                            format!("Mandatory feature {} is missing required metric {}.", feature.feature_ref, metric.metric_id));
                    }
                }
            }
        }

        if self.state == "adjudicated" {
            match self.engineering_review.as_ref() {
                Some(review) => {
                    if is_blank(&review.reviewer_ref) || is_blank(&review.reviewed_at) || is_blank(&review.review_record_ref) {
                        push(&mut findings, "engineering_review_identity_missing", "engineering_review",
                            "Adjudication requires reviewer, timestamp and review-record references.");
                    }
                    if !matches!(review.disposition.as_str(), "accept" | "reject" | "hold") {
                        push(&mut findings, "invalid_engineering_disposition", "engineering_review.disposition",
                            "Engineering disposition must be accept, reject or hold.");
                    }
                    let any_fail = assessments.iter().any(|item| item.disposition == "fail");
                    let any_blocked = assessments.iter().any(|item|
                        item.disposition == "uncertainty_blocked" || item.disposition == "insufficient_evidence");
                    if review.disposition == "accept" && (any_fail || any_blocked) {
                        push(&mut findings, "acceptance_with_nonpassing_measurements", "engineering_review.disposition",
                            "As-built acceptance requires every mandatory measured feature to pass guard-banded limits.");
                    }
                    if review.disposition == "accept" && self.observations.len() == 0 {
                        push(&mut findings, "acceptance_without_measurements", "observations",
                            "An as-built inspection cannot be accepted without physical measurements.");
                    }
                }
                None => push(&mut findings, "engineering_review_missing", "engineering_review",
                    "Adjudicated inspection requires an explicit engineering review record."),
            }
        } else if self.engineering_review.is_some() {
            push(&mut findings, "review_attached_before_adjudication", "engineering_review",
                "Engineering review is only attached at the adjudicated state.");
        }

        if !self.evidence_policy.physical_measurements_require_current_calibration
            || !self.evidence_policy.datum_registration_must_exclude_test_roi
            || !self.evidence_policy.independent_holdout_datums_required
            || !self.evidence_policy.registration_uncertainty_included_in_measurement_budget
            || !self.evidence_policy.missing_coverage_never_passes
            || !self.evidence_policy.guard_banded_acceptance_required
            || !self.evidence_policy.engineering_adjudication_required
            || !self.evidence_policy.as_built_acceptance_is_not_product_release
        {
            push(&mut findings, "unsafe_as_built_policy", "evidence_policy",
                "Calibration, independent datums, test-ROI exclusion, uncertainty, coverage and adjudication boundaries are mandatory.");
        }

        let pass_count = assessments.iter().filter(|item| item.disposition == "pass").count();
        let fail_count = assessments.iter().filter(|item| item.disposition == "fail").count();
        let uncertainty_blocked_count = assessments.iter().filter(|item| item.disposition == "uncertainty_blocked").count();
        let insufficient_evidence_count = assessments.iter().filter(|item| item.disposition == "insufficient_evidence").count();
        let assessed_observation_count = assessments.len();
        let as_built_acceptance_admitted = self.state == "adjudicated"
            && self.engineering_review.as_ref().is_some_and(|review| review.disposition == "accept")
            && findings.is_empty()
            && self.feature_catalog.iter().filter(|feature| feature.mandatory).all(|feature|
                feature.required_metrics.iter().all(|metric| {
                    if metric.metric_id == "joint_clearance_mm" {
                        feature.required_pose_refs.iter().all(|pose_ref| assessments.iter().any(|assessment|
                            assessment.feature_ref == feature.feature_ref
                                && assessment.metric_id == metric.metric_id
                                && assessment.pose_ref.as_deref() == Some(pose_ref.as_str())
                                && assessment.disposition == "pass"))
                    } else {
                        assessments.iter().any(|assessment|
                            assessment.feature_ref == feature.feature_ref
                                && assessment.metric_id == metric.metric_id
                                && assessment.disposition == "pass")
                    }
                }));

        ValidationReport {
            findings,
            feature_count: self.feature_catalog.len(),
            observation_count: self.observations.len(),
            assessed_observation_count,
            pass_count,
            fail_count,
            uncertainty_blocked_count,
            insufficient_evidence_count,
            feature_assessments: assessments,
            as_built_acceptance_admitted,
        }
    }
}

fn validate_article_binding(article: &ArticleBinding, state: &str, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("candidate_id", article.candidate_id.as_str()),
        ("manufactured_article_ref", article.manufactured_article_ref.as_str()),
        ("component_revision_ref", article.component_revision_ref.as_str()),
        ("route_ref", article.route_ref.as_str()),
        ("route_revision_ref", article.route_revision_ref.as_str()),
        ("topology_ref", article.topology_ref.as_str()),
        ("topology_revision_ref", article.topology_revision_ref.as_str()),
        ("robot_id", article.robot_id.as_str()),
        ("robot_article_revision", article.robot_article_revision.as_str()),
        ("material_lot_ref", article.material_lot_ref.as_str()),
        ("process_revision_ref", article.process_revision_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "article_binding_missing", format!("article.{field}"),
                "As-built article binding requires exact non-empty revision identity.");
        }
    }
    for (field, value) in [
        ("route_digest", article.route_digest.as_str()),
        ("topology_digest", article.topology_digest.as_str()),
        ("geometry_digest", article.geometry_digest.as_str()),
    ] {
        if !is_sha256_digest(value) {
            push(findings, "article_digest_invalid", format!("article.{field}"),
                "Article, route, topology and geometry digests must use sha256: followed by 64 hexadecimal characters.");
        }
    }
    if !matches!(article.treatment_state.as_str(), "as_manufactured" | "after_cleaning" | "after_flex" | "after_mission_replay" | "after_repair") {
        push(findings, "invalid_treatment_state", "article.treatment_state",
            "Unknown as-built article treatment/conditioning state.");
    }
    if state != "proposed" && is_blank(&article.manufactured_article_ref) {
        push(findings, "physical_article_identity_missing", "article.manufactured_article_ref",
            "Measured inspection must bind the exact manufactured article/component.");
    }
}

fn validate_session(
    session: &MeasurementSession,
    features: &HashMap<&str, &FeatureSpec>,
    observations: &[FeatureMeasurement],
    findings: &mut Vec<Finding>,
) {
    for (field, value) in [
        ("session_id", session.session_id.as_str()),
        ("instrument_ref", session.instrument_ref.as_str()),
        ("calibration_ref", session.calibration_ref.as_str()),
        ("calibration_revision_ref", session.calibration_revision_ref.as_str()),
        ("method_ref", session.method_ref.as_str()),
        ("protocol_ref", session.protocol_ref.as_str()),
        ("fixture_ref", session.fixture_ref.as_str()),
        ("fixture_revision_ref", session.fixture_revision_ref.as_str()),
        ("reference_artifact_ref", session.reference_artifact_ref.as_str()),
        ("environment_ref", session.environment_ref.as_str()),
        ("start_time", session.start_time.as_str()),
        ("end_time", session.end_time.as_str()),
        ("parser_revision_ref", session.parser_revision_ref.as_str()),
        ("coordinate_frame_ref", session.coordinate_frame_ref.as_str()),
        ("datum_scheme_ref", session.datum_scheme_ref.as_str()),
        ("transform_ref", session.transform_ref.as_str()),
        ("registration_evidence_ref", session.registration_evidence_ref.as_str()),
        ("uncertainty_budget_ref", session.uncertainty_budget_ref.as_str()),
        ("operator_ref", session.operator_ref.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "measurement_session_binding_missing", format!("measurement_session.{field}"),
                "A measured session requires exact instrument, calibration, method, fixture, environment, time, registration and operator provenance.");
        }
    }
    if !matches!(session.instrument_kind.as_str(), "structured_light_3d" | "laser_3d" | "photogrammetry" | "contact_probe" | "dimensional_fixture") {
        push(findings, "unknown_metrology_instrument_kind", "measurement_session.instrument_kind",
            "Unknown as-built metrology instrument kind.");
    }
    if session.calibration_state != "valid" {
        push(findings, "instrument_calibration_not_current", "measurement_session.calibration_state",
            "Physical as-built measurements require an explicitly current/valid instrument calibration.");
    }
    if !is_sha256_digest(&session.raw_data_digest) {
        push(findings, "measurement_raw_digest_invalid", "measurement_session.raw_data_digest",
            "Raw point-cloud/measurement data digest must use sha256: followed by 64 hexadecimal characters.");
    }
    if !matches!(session.registration_method.as_str(), "datum_based_locked" | "fiducial_based_locked" | "validated_transform_chain") {
        push(findings, "registration_method_not_admitted", "measurement_session.registration_method",
            "Unconstrained best-fit/ICP registration is not admitted for acceptance; use locked datum/fiducial or a validated transform chain.");
    }
    if session.fit_datum_refs.is_empty() || session.fit_datum_refs.iter().any(|datum| is_blank(datum)) {
        push(findings, "registration_fit_datums_missing", "measurement_session.fit_datum_refs",
            "Registration requires non-empty explicit datum identities used to compute the transform.");
    }
    if session.holdout_datum_refs.is_empty() || session.holdout_datum_refs.iter().any(|datum| is_blank(datum)) {
        push(findings, "independent_holdout_datums_missing", "measurement_session.holdout_datum_refs",
            "Registration requires at least one non-empty independent hold-out datum not used to compute the transform.");
    }
    let fit_datums: HashSet<&str> = session.fit_datum_refs.iter().map(String::as_str).collect();
    if fit_datums.len() != session.fit_datum_refs.len() {
        push(findings, "duplicate_registration_fit_datum", "measurement_session.fit_datum_refs",
            "Registration-fit datum identities must be unique.");
    }
    let holdout_datums: HashSet<&str> = session.holdout_datum_refs.iter().map(String::as_str).collect();
    if holdout_datums.len() != session.holdout_datum_refs.len() {
        push(findings, "duplicate_registration_holdout_datum", "measurement_session.holdout_datum_refs",
            "Independent hold-out datum identities must be unique.");
    }
    if !fit_datums.is_disjoint(&holdout_datums) {
        push(findings, "registration_fit_holdout_overlap", "measurement_session.fit_datum_refs/holdout_datum_refs",
            "A datum used to fit registration cannot also serve as an independent hold-out datum.");
    }
    let mut residual_refs = HashSet::new();
    if session.holdout_datum_residuals.len() != session.holdout_datum_refs.len() {
        push(findings, "holdout_residual_coverage_incomplete", "measurement_session.holdout_datum_residuals",
            "Every independent hold-out datum requires its own residual and uncertainty record.");
    }
    let mut maximum_observed_residual = 0.0_f64;
    for (index, residual) in session.holdout_datum_residuals.iter().enumerate() {
        if !holdout_datums.contains(residual.datum_ref.as_str()) {
            push(findings, "residual_for_unlisted_holdout_datum", format!("measurement_session.holdout_datum_residuals[{index}].datum_ref"),
                "Residual datum must be declared in the independent hold-out set.");
        }
        if !residual_refs.insert(residual.datum_ref.as_str()) {
            push(findings, "duplicate_holdout_datum_residual", format!("measurement_session.holdout_datum_residuals[{index}].datum_ref"),
                "Each hold-out datum must have exactly one residual record.");
        }
        if fit_datums.contains(residual.datum_ref.as_str()) {
            push(findings, "holdout_datum_was_used_for_fit", format!("measurement_session.holdout_datum_residuals[{index}].datum_ref"),
                "A hold-out datum cannot be included in the registration fit set.");
        }
        if !residual.residual_mm.is_finite() || residual.residual_mm < 0.0
            || !residual.expanded_uncertainty_mm.is_finite() || residual.expanded_uncertainty_mm < 0.0
        {
            push(findings, "invalid_holdout_datum_residual", format!("measurement_session.holdout_datum_residuals[{index}]"),
                "Per-datum residual and expanded uncertainty must be finite and non-negative.");
        } else {
            maximum_observed_residual = maximum_observed_residual.max(residual.residual_mm);
            if residual.residual_mm + residual.expanded_uncertainty_mm > session.maximum_holdout_residual_mm {
                push(findings, "holdout_residual_guard_band_exceeded", format!("measurement_session.holdout_datum_residuals[{index}]"),
                    "Per-datum residual plus its expanded uncertainty exceeds the maximum registration residual.");
            }
        }
    }
    if !session.holdout_datum_residuals.is_empty()
        && (session.observed_holdout_residual_mm - maximum_observed_residual).abs() > 1e-9
    {
        push(findings, "aggregate_holdout_residual_mismatch", "measurement_session.observed_holdout_residual_mm",
            "Aggregate hold-out residual must equal the maximum of the per-datum residual records.");
    }
    let mut holdout_ids = HashSet::new();
    for datum in &session.holdout_datum_refs {
        if is_blank(datum) || !holdout_ids.insert(datum.as_str()) {
            push(findings, "duplicate_or_blank_holdout_datum", "measurement_session.holdout_datum_refs",
                "Hold-out datum identities must be non-empty and unique.");
        }
    }
    if session.excluded_roi_feature_refs.is_empty() {
        push(findings, "registration_roi_exclusion_missing", "measurement_session.excluded_roi_feature_refs",
            "Registration must explicitly exclude inspected feature ROIs from the alignment fit.");
    }
    let excluded_rois: HashSet<&str> = session.excluded_roi_feature_refs.iter().map(String::as_str).collect();
    if excluded_rois.len() != session.excluded_roi_feature_refs.len() {
        push(findings, "duplicate_registration_roi_exclusion", "measurement_session.excluded_roi_feature_refs",
            "Registration ROI exclusion references must be unique.");
    }
    for feature in features.values().filter(|feature| feature.mandatory) {
        if !excluded_rois.contains(feature.feature_ref) {
            push(findings, "mandatory_feature_not_excluded_from_registration", "measurement_session.excluded_roi_feature_refs",
                format!("Mandatory inspection feature {} must be excluded from registration fitting.", feature.feature_ref));
        }
    }
    for feature_ref in &session.excluded_roi_feature_refs {
        if !features.contains_key(feature_ref.as_str()) {
            push(findings, "registration_roi_exclusion_unknown_feature", "measurement_session.excluded_roi_feature_refs",
                format!("Registration exclusion references unknown feature {feature_ref}."));
        }
    }
    if !session.maximum_holdout_residual_mm.is_finite()
        || session.maximum_holdout_residual_mm <= 0.0
        || !session.observed_holdout_residual_mm.is_finite()
        || session.observed_holdout_residual_mm < 0.0
        || !session.registration_uncertainty_mm.is_finite()
        || session.registration_uncertainty_mm < 0.0
    {
        push(findings, "invalid_registration_uncertainty_or_residual", "measurement_session",
            "Registration residual limits, observed residual and registration uncertainty must be finite and non-negative.");
    } else if session.observed_holdout_residual_mm > session.maximum_holdout_residual_mm {
        push(findings, "registration_holdout_residual_exceeded", "measurement_session.observed_holdout_residual_mm",
            "Independent hold-out datum residual exceeds the registered measurement profile limit.");
    }
    if !session.uncertainty_budget_includes_registration {
        push(findings, "registration_uncertainty_omitted", "measurement_session.uncertainty_budget_includes_registration",
            "The per-feature expanded uncertainty budget must include registration uncertainty.");
    }
    for (field, value) in [
        ("coverage_fraction", session.coverage_fraction),
        ("occlusion_fraction", session.occlusion_fraction),
        ("missing_data_fraction", session.missing_data_fraction),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            push(findings, "invalid_session_coverage_fraction", format!("measurement_session.{field}"),
                "Session coverage, occlusion and missing-data fractions must be within 0..=1.");
        }
    }
    if session.missing_data_fraction > 0.0 || session.occlusion_fraction > 0.0 {
        if is_blank_option(&session.gap_disposition_ref) {
            push(findings, "measurement_gap_disposition_missing", "measurement_session.gap_disposition_ref",
                "Occluded or missing regions require an explicit disposition; interpolation cannot silently create a pass.");
        }
    }
    let mut excluded_ids = HashSet::new();
    for feature_ref in &session.excluded_roi_feature_refs {
        if !features.contains_key(feature_ref.as_str()) {
            push(findings, "registration_excludes_unknown_feature", "measurement_session.excluded_roi_feature_refs",
                format!("Registration exclusion references unknown feature {feature_ref}."));
        }
        if !excluded_ids.insert(feature_ref.as_str()) {
            push(findings, "duplicate_registration_exclusion", "measurement_session.excluded_roi_feature_refs",
                "Registration ROI exclusions must be unique.");
        }
        if session.holdout_datum_refs.iter().any(|datum| datum == feature_ref) {
            push(findings, "holdout_datum_overlaps_test_roi", "measurement_session.holdout_datum_refs",
                "Independent hold-out datums cannot be one of the inspected feature ROIs.");
        }
    }
    for observation in observations {
        if !session.excluded_roi_feature_refs.iter().any(|feature_ref| feature_ref == &observation.feature_ref) {
            push(findings, "test_roi_not_excluded_from_registration", "measurement_session.excluded_roi_feature_refs",
                format!("Inspected feature {} is not explicitly excluded from registration fitting.", observation.feature_ref));
        }
        if let Some(feature) = features.get(observation.feature_ref.as_str()) {
            if session.coverage_fraction < feature.minimum_coverage_fraction {
                push(findings, "session_coverage_below_feature_requirement", "measurement_session.coverage_fraction",
                    "Session-wide coverage is below this feature's minimum coverage requirement.");
            }
        } else {
            push(findings, "session_observation_unknown_feature", "observations.feature_ref",
                "Measurement feature is not part of the nominal feature catalog.");
        }
    }
}

fn metric_definition(metric: &str) -> Option<&'static str> {
    match metric {
        "panel_surface_deviation_mm" => Some("mm"),
        "seam_path_deviation_mm" => Some("mm"),
        "edge_length_error_mm" => Some("mm"),
        "service_opening_position_error_mm" => Some("mm"),
        "attachment_position_error_mm" => Some("mm"),
        "joint_clearance_mm" => Some("mm"),
        "garment_landmark_deviation_mm" => Some("mm"),
        "post_cleaning_dimensional_change_pct" => Some("%"),
        _ => None,
    }
}

fn metric_unit_matches(metric: &str, unit: &str) -> bool {
    metric_definition(metric).is_some_and(|expected| expected == unit)
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

    fn base_manifest() -> Value {
        json!({
            "schema_version":"0.1.0",
            "inspection_id":"LIGHTSHELL-ASBUILT-SYNTHETIC-001",
            "state":"proposed",
            "article":{
                "candidate_id":"ELBOW-CANDIDATE-001",
                "manufactured_article_ref":"SYNTHETIC-ARTICLE-001",
                "component_revision_ref":"GARMENT-COMPONENT-REV-A",
                "route_ref":"LIGHTSHELL-ROUTE-12800",
                "route_revision_ref":"ROUTE-REV-A",
                "route_digest":format!("sha256:{}", "a".repeat(64)),
                "topology_ref":"LIGHTSHELL-TOPOLOGY-12785",
                "topology_revision_ref":"TOPOLOGY-REV-A",
                "topology_digest":format!("sha256:{}", "b".repeat(64)),
                "robot_id":"SYNTHETIC-ROBOT-001",
                "robot_article_revision":"ROBOT-REV-A",
                "geometry_digest":format!("sha256:{}", "c".repeat(64)),
                "material_lot_ref":"LOT-001",
                "process_revision_ref":"PROCESS-REV-A",
                "treatment_state":"as_manufactured"
            },
            "feature_catalog":[
                {"feature_ref":"forearm_panel","feature_kind":"panel","nominal_geometry_ref":"CAD-PANEL-001","nominal_revision_ref":"CAD-REV-A","mandatory":true,"required_metrics":[{"metric_id":"panel_surface_deviation_mm","unit":"mm","lower_limit":0.0,"upper_limit":1.0}],"required_pose_refs":[],"minimum_coverage_fraction":0.95,"acceptance_rule_ref":"PANEL-ACCEPTANCE-01"},
                {"feature_ref":"elbow_seam","feature_kind":"connection","nominal_geometry_ref":"SEAM-CURVE-001","nominal_revision_ref":"TOPOLOGY-REV-A","mandatory":true,"required_metrics":[{"metric_id":"seam_path_deviation_mm","unit":"mm","lower_limit":0.0,"upper_limit":2.0}],"required_pose_refs":[],"minimum_coverage_fraction":0.98,"acceptance_rule_ref":"SEAM-ACCEPTANCE-01"}
            ],
            "measurement_session":null,
            "observations":[],
            "engineering_review":null,
            "evidence_policy":{
                "physical_measurements_require_current_calibration":true,
                "datum_registration_must_exclude_test_roi":true,
                "independent_holdout_datums_required":true,
                "registration_uncertainty_included_in_measurement_budget":true,
                "missing_coverage_never_passes":true,
                "guard_banded_acceptance_required":true,
                "engineering_adjudication_required":true,
                "as_built_acceptance_is_not_product_release":true
            }
        })
    }

    fn complete_session() -> Value {
        json!({
            "session_id":"SCAN-SESSION-001",
            "instrument_kind":"structured_light_3d",
            "instrument_ref":"SCANNER-SERIAL-001",
            "calibration_ref":"CAL-CERT-001",
            "calibration_revision_ref":"CAL-REV-A",
            "calibration_state":"valid",
            "method_ref":"ASBUILT-METHOD-01",
            "protocol_ref":"SCAN-PROTOCOL-REV-A",
            "fixture_ref":"ROBOT-ARTICLE-FIXTURE-01",
            "fixture_revision_ref":"FIXTURE-REV-A",
            "reference_artifact_ref":"SCALE-ARTIFACT-01",
            "environment_ref":"ENV-RECORD-01",
            "start_time":"2026-11-18T10:00:00+02:00",
            "end_time":"2026-11-18T10:15:00+02:00",
            "raw_data_digest":format!("sha256:{}", "d".repeat(64)),
            "parser_revision_ref":"PARSER-REV-A",
            "coordinate_frame_ref":"ROBOT-ROOT-REV-A",
            "registration_method":"datum_based_locked",
            "datum_scheme_ref":"DATUM-SCHEME-REV-A",
            "transform_ref":"LOCKED-TRANSFORM-01",
            "registration_evidence_ref":"REGISTRATION-REPORT-01",
            "fit_datum_refs":["FIT-DATUM-01","FIT-DATUM-02","FIT-DATUM-03"],
            "holdout_datum_refs":["HOLDOUT-DATUM-01","HOLDOUT-DATUM-02"],
            "holdout_datum_residuals":[
                {"datum_ref":"HOLDOUT-DATUM-01","residual_mm":0.15,"expanded_uncertainty_mm":0.05},
                {"datum_ref":"HOLDOUT-DATUM-02","residual_mm":0.2,"expanded_uncertainty_mm":0.05}
            ],
            "excluded_roi_feature_refs":["forearm_panel","elbow_seam"],
            "maximum_holdout_residual_mm":0.5,
            "observed_holdout_residual_mm":0.2,
            "registration_uncertainty_mm":0.15,
            "uncertainty_budget_ref":"UQ-BUDGET-01",
            "uncertainty_budget_includes_registration":true,
            "coverage_fraction":0.99,
            "occlusion_fraction":0.01,
            "missing_data_fraction":0.01,
            "gap_disposition_ref":"OCCLUSION-BOUND-01",
            "operator_ref":"OPERATOR-01"
        })
    }

    fn measured_observation(id: &str, feature: &str, metric: &str, value: f64, uncertainty: f64, upper: f64) -> Value {
        json!({
            "measurement_id":id,"feature_ref":feature,"metric_id":metric,"unit":"mm",
            "measured_value":value,"expanded_uncertainty":uncertainty,"lower_limit":0.0,"upper_limit":upper,
            "coverage_fraction":0.99,"measurement_origin":"physical_measurement",
            "method_selection_ref":"METHOD-SELECTION-01","raw_data_slice_ref":format!("RAW-SLICE-{id}"),
            "uncertainty_budget_ref":"UQ-BUDGET-01","measurement_evidence_ref":format!("MEAS-EVIDENCE-{id}")
        })
    }

    fn make_measured(manifest: &mut Value) {
        manifest["state"] = json!("measured");
        manifest["measurement_session"] = complete_session();
        manifest["observations"] = json!([
            measured_observation("MEAS-PANEL-01","forearm_panel","panel_surface_deviation_mm",0.8,0.1,1.0),
            measured_observation("MEAS-SEAM-01","elbow_seam","seam_path_deviation_mm",1.4,0.2,2.0)
        ]);
    }

    #[test]
    fn proposed_asbuilt_inspection_fixture_is_valid_without_physical_claim() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.feature_count, 2);
        assert!(!report.can_accept_as_built());
    }

    #[test]
    fn measured_guard_banded_features_can_be_accepted_after_engineering_review() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["state"] = json!("adjudicated");
        manifest["engineering_review"] = json!({
            "disposition":"accept","reviewer_ref":"ENGINEER-01",
            "reviewed_at":"2026-11-18T11:00:00+02:00","review_record_ref":"REVIEW-RECORD-01"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.pass_count, 2);
        assert!(report.can_accept_as_built());
    }

    #[test]
    fn asbuilt_decision_requires_at_least_one_mandatory_feature() {
        let mut manifest = base_manifest();
        for feature in manifest["feature_catalog"].as_array_mut().unwrap() {
            feature["mandatory"] = json!(false);
        }
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "no_mandatory_asbuilt_features"));
    }

    #[test]
    fn independent_holdout_datum_cannot_be_an_inspected_roi() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["holdout_datum_refs"] = json!(["forearm_panel"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "holdout_datum_overlaps_test_roi"));
    }

    #[test]
    fn session_wide_coverage_must_meet_each_feature_requirement() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["coverage_fraction"] = json!(0.90);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "session_coverage_below_feature_requirement"));
    }

    #[test]
    fn best_fit_registration_is_not_admitted_for_asbuilt_acceptance() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["registration_method"] = json!("unconstrained_best_fit_icp");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "registration_method_not_admitted"));
    }

    #[test]
    fn inspected_roi_must_be_excluded_from_registration_fit() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["excluded_roi_feature_refs"] = json!(["forearm_panel"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "test_roi_not_excluded_from_registration"));
    }

    #[test]
    fn independent_holdout_datums_are_required() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["holdout_datum_refs"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "independent_holdout_datums_missing"));
    }

    #[test]
    fn registration_holdout_residual_must_remain_within_profile_limit() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["observed_holdout_residual_mm"] = json!(0.8);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "registration_holdout_residual_exceeded"));
    }

    #[test]
    fn registration_uncertainty_must_be_included_in_feature_budget() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["uncertainty_budget_includes_registration"] = json!(false);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "registration_uncertainty_omitted"));
    }

    #[test]
    fn expired_or_unknown_instrument_calibration_fails_closed() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["calibration_state"] = json!("expired");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "instrument_calibration_not_current"));
    }

    #[test]
    fn raw_point_cloud_digest_is_mandatory() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["raw_data_digest"] = json!("not-a-digest");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "measurement_raw_digest_invalid"));
    }

    #[test]
    fn uncertainty_interval_crossing_tolerance_is_blocked() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["observations"][0]["measured_value"] = json!(0.95);
        manifest["observations"][0]["expanded_uncertainty"] = json!(0.1);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert_eq!(report.feature_assessments[0].disposition, "uncertainty_blocked");
    }

    #[test]
    fn measurement_interval_entirely_outside_tolerance_fails() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["observations"][0]["measured_value"] = json!(1.5);
        manifest["observations"][0]["expanded_uncertainty"] = json!(0.1);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert_eq!(report.feature_assessments[0].disposition, "fail");
    }

    #[test]
    fn low_coverage_is_insufficient_evidence_not_pass() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["observations"][0]["coverage_fraction"] = json!(0.8);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert_eq!(report.feature_assessments[0].disposition, "insufficient_evidence");
        assert!(report.findings.iter().any(|f| f.code == "feature_coverage_below_requirement"));
    }

    #[test]
    fn missing_mandatory_feature_metric_blocks_measured_inspection() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["observations"].as_array_mut().unwrap().pop();
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "mandatory_feature_measurement_missing"));
    }

    #[test]
    fn synthetic_or_supplier_data_cannot_be_promoted_to_measured_state() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["observations"][0]["measurement_origin"] = json!("supplier_claim");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "nonphysical_measurement_promoted"));
    }

    #[test]
    fn article_route_topology_and_geometry_digests_are_required() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["article"]["topology_digest"] = json!("bad");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "article_digest_invalid"));
    }

    #[test]
    fn missing_data_and_occlusion_require_explicit_disposition() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["gap_disposition_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "measurement_gap_disposition_missing"));
    }

    #[test]
    fn adjudication_cannot_accept_a_blocked_measurement() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["state"] = json!("adjudicated");
        manifest["observations"][0]["measured_value"] = json!(0.95);
        manifest["observations"][0]["expanded_uncertainty"] = json!(0.1);
        manifest["engineering_review"] = json!({
            "disposition":"accept","reviewer_ref":"ENGINEER-01",
            "reviewed_at":"2026-11-18T11:00:00+02:00","review_record_ref":"REVIEW-RECORD-01"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "acceptance_with_nonpassing_measurements"));
        assert!(!report.can_accept_as_built());
    }

    #[test]
    fn mandatory_feature_must_be_excluded_from_registration_even_when_no_observation_exists() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["observations"].as_array_mut().unwrap().pop();
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "mandatory_feature_measurement_missing"));
    }

    #[test]
    fn duplicate_feature_metric_measurements_are_rejected() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        let duplicate = manifest["observations"][0].clone();
        manifest["observations"].as_array_mut().unwrap().push(duplicate);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_feature_metric_measurement"));
    }

    #[test]
    fn post_cleaning_dimensional_change_uses_percent_units() {
        let mut manifest = base_manifest();
        manifest["feature_catalog"][0]["required_metrics"] = json!([{
            "metric_id":"post_cleaning_dimensional_change_pct","unit":"%","lower_limit":-2.0,"upper_limit":2.0
        }]);
        make_measured(&mut manifest);
        manifest["observations"][0]["metric_id"] = json!("post_cleaning_dimensional_change_pct");
        manifest["observations"][0]["unit"] = json!("%");
        manifest["observations"][0]["measured_value"] = json!(1.2);
        manifest["observations"][0]["lower_limit"] = json!(-2.0);
        manifest["observations"][0]["upper_limit"] = json!(2.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
    }

    #[test]
    fn observation_cannot_widen_frozen_feature_acceptance_limits() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["observations"][0]["upper_limit"] = json!(5.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "measurement_acceptance_rule_mismatch"));
    }

    #[test]
    fn registration_fit_and_holdout_datums_must_be_disjoint() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["fit_datum_refs"] = json!(["FIT-DATUM-01","HOLDOUT-DATUM-01"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "registration_fit_holdout_overlap"));
    }

    #[test]
    fn every_holdout_datum_requires_its_own_residual_record() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["holdout_datum_residuals"].as_array_mut().unwrap().pop();
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "holdout_residual_coverage_incomplete"));
    }

    #[test]
    fn holdout_residual_plus_uncertainty_must_fit_registration_limit() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["holdout_datum_residuals"][0]["residual_mm"] = json!(0.46);
        manifest["measurement_session"]["holdout_datum_residuals"][0]["expanded_uncertainty_mm"] = json!(0.06);
        manifest["measurement_session"]["observed_holdout_residual_mm"] = json!(0.46);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "holdout_residual_guard_band_exceeded"));
    }

    #[test]
    fn every_mandatory_feature_is_excluded_from_registration_fit() {
        let mut manifest = base_manifest();
        make_measured(&mut manifest);
        manifest["measurement_session"]["excluded_roi_feature_refs"] = json!(["forearm_panel"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "mandatory_feature_not_excluded_from_registration"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["invented_scan_confidence"] = json!(0.999);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
