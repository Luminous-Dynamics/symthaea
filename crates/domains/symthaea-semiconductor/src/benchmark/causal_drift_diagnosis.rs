// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Falsifiable causal-drift hypothesis assessment for semiconductor VM.
//!
//! The module separates observational association from controlled-intervention
//! consistency. Feature attribution and temporal co-movement cannot produce a
//! causal status. Matched difference-in-differences remains association-only
//! because balance and parallel-trend diagnostics cannot rule out hidden
//! confounding. No physical experiment or equipment action is executed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const CAUSAL_DRIFT_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-drift-input.v1";
pub const CAUSAL_DRIFT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-drift-report.v1";
pub const CAUSAL_DRIFT_EVIDENCE_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-drift-evidence.v1";

const NORMAL_95_Z: f64 = 1.959_963_984_540_054;
/// Conservative 95% Student-t critical value for df >= 9 (10 independent clusters).
const T_95_DF9: f64 = 2.262_157_162_740_992;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDriftHypothesisKindV1 {
    SensorDrift,
    CalibrationDrift,
    ToolStateChange,
    ChamberLocalChange,
    RecipeOrProcessChange,
    FacilityCommonCause,
    MaterialOrProductMix,
    TimeAlignmentOrJoin,
    SamplingOrSelectionBias,
    ModelOrPreprocessingChange,
    UnknownOrUnmodelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedEffectDirectionV1 {
    Positive,
    Negative,
    NonZero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDriftEvidenceKindV1 {
    FeatureAttribution,
    TemporalPrecedence,
    MaintenanceRecord,
    ReferenceMeasurement,
    ClockJoinAudit,
    SourceInventory,
    MatchedContrast,
    ControlledIntervention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalAssessmentStatusV1 {
    AssociationOnly,
    InterventionConsistent,
    Contradicted,
    Unidentifiable,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftEvidenceV1 {
    pub schema: String,
    pub evidence_id: String,
    pub evidence_kind: CausalDriftEvidenceKindV1,
    pub hypothesis_id: String,
    pub event_time_unix_seconds: i64,
    pub available_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub source_artifact_sha256: String,
    pub authority_ref: String,
    /// A contribution/association score is not a causal effect estimate.
    pub attribution_score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchedDifferenceInDifferencesV1 {
    pub treated_pre_mean: f64,
    pub treated_post_mean: f64,
    pub control_pre_mean: f64,
    pub control_post_mean: f64,
    /// Sample variances within each group-period cell.
    pub treated_pre_variance: f64,
    pub treated_post_variance: f64,
    pub control_pre_variance: f64,
    pub control_post_variance: f64,
    pub treated_pre_count: usize,
    pub treated_post_count: usize,
    pub control_pre_count: usize,
    pub control_post_count: usize,
    pub maximum_absolute_covariate_smd: f64,
    pub treated_pretrend_slope: f64,
    pub control_pretrend_slope: f64,
    pub protocol_sha256: String,
    pub source_summary_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterDidSummaryV1 {
    pub cluster_id: String,
    pub treated_pre_mean: f64,
    pub treated_post_mean: f64,
    pub control_pre_mean: f64,
    pub control_post_mean: f64,
    pub treated_pre_count: usize,
    pub treated_post_count: usize,
    pub control_pre_count: usize,
    pub control_post_count: usize,
    pub maximum_absolute_covariate_smd: f64,
    pub treated_pretrend_slope: f64,
    pub control_pretrend_slope: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusteredDifferenceInDifferencesV1 {
    /// Each row is one independent top-level unit (e.g. tool, chamber or lot).
    /// Units nested within a cluster are summarized before this estimator.
    pub clusters: Vec<ClusterDidSummaryV1>,
    pub protocol_sha256: String,
    pub source_summary_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlledInterventionEvidenceV1 {
    pub randomized_assignment: bool,
    pub preregistered_analysis: bool,
    pub control_group_present: bool,
    pub manipulation_check_passed: bool,
    pub interference_assumption_declared: bool,
    pub treated_count: usize,
    pub control_count: usize,
    pub effect_estimate: f64,
    pub standard_error: f64,
    pub assignment_protocol_sha256: String,
    pub preregistration_sha256: String,
    pub independent_verification_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NegativeControlEvidenceV1 {
    pub control_id: String,
    pub source_artifact_sha256: String,
    pub effect_estimate: f64,
    pub standard_error: f64,
    /// Maximum effect magnitude considered practically negligible.
    pub materiality_margin: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftHypothesisInputV1 {
    pub hypothesis_id: String,
    pub hypothesis: CausalDriftHypothesisKindV1,
    pub expected_effect_direction: ExpectedEffectDirectionV1,
    pub scope: String,
    pub evidence: Vec<CausalDriftEvidenceV1>,
    pub matched_contrast: Option<MatchedDifferenceInDifferencesV1>,
    pub clustered_matched_contrast: Option<ClusteredDifferenceInDifferencesV1>,
    pub controlled_intervention: Option<ControlledInterventionEvidenceV1>,
    pub negative_controls: Vec<NegativeControlEvidenceV1>,
    pub measured_confounders: Vec<String>,
    pub unmeasured_confounders: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftPolicyV1 {
    pub minimum_group_count: usize,
    pub maximum_covariate_smd: f64,
    pub maximum_pretrend_slope_difference: f64,
    /// Minimum number of independent top-level clusters for cluster-level DiD.
    pub minimum_independent_clusters: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftDiagnosisInputV1 {
    pub schema: String,
    pub incident_id: String,
    pub incident_clock_domain_id: String,
    pub incident_time_unix_seconds: i64,
    pub analysis_cutoff_unix_seconds: i64,
    pub source_inventory_complete: bool,
    pub source_inventory_sha256: String,
    pub clock_join_integrity: bool,
    pub clock_join_audit_sha256: String,
    pub policy: CausalDriftPolicyV1,
    pub hypotheses: Vec<CausalDriftHypothesisInputV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalEffectEstimateV1 {
    pub estimate: f64,
    pub standard_error: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub method: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalHypothesisAssessmentV1 {
    pub hypothesis_id: String,
    pub hypothesis: CausalDriftHypothesisKindV1,
    pub scope: String,
    pub status: CausalAssessmentStatusV1,
    pub reasons: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub observational_evidence_count: usize,
    pub matched_contrast: Option<CausalEffectEstimateV1>,
    pub controlled_intervention: Option<CausalEffectEstimateV1>,
    pub negative_control_violation_ids: Vec<String>,
    pub measured_confounders: Vec<String>,
    pub unmeasured_confounders: Vec<String>,
    pub causal_claim_authorized: bool,
    pub human_review_required: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftDiagnosisReportV1 {
    pub schema: String,
    pub incident_id: String,
    pub analysis_cutoff_unix_seconds: i64,
    pub input_sha256: String,
    pub status_by_hypothesis: Vec<CausalHypothesisAssessmentV1>,
    pub source_authenticity_proven: bool,
    pub causal_claim_authorized: bool,
    pub physical_action_authorized: bool,
    pub equipment_control_authority: bool,
    pub claim_ceiling: Vec<String>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CausalDriftDiagnosisErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), CausalDriftDiagnosisErrorV1> {
    if value.trim().is_empty() {
        return Err(CausalDriftDiagnosisErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, CausalDriftDiagnosisErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CausalDriftDiagnosisErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn input_digest(
    input: &CausalDriftDiagnosisInputV1,
) -> Result<String, CausalDriftDiagnosisErrorV1> {
    let mut canonical = input.clone();
    canonical.hypotheses.sort_by(|left, right| left.hypothesis_id.cmp(&right.hypothesis_id));
    for hypothesis in &mut canonical.hypotheses {
        hypothesis.evidence.sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));
        hypothesis.negative_controls.sort_by(|left, right| left.control_id.cmp(&right.control_id));
        hypothesis.measured_confounders.sort();
        hypothesis.unmeasured_confounders.sort();
    }
    json_sha256(&canonical)
}

fn report_digest(
    report: &CausalDriftDiagnosisReportV1,
) -> Result<String, CausalDriftDiagnosisErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Returns a content digest over the canonicalized input contract. It is not a
/// signature and does not establish the authenticity of the source evidence.
pub fn seal_causal_drift_diagnosis_input_v1(
    input: &CausalDriftDiagnosisInputV1,
) -> Result<String, CausalDriftDiagnosisErrorV1> {
    input_digest(input)
}

fn estimate_interval(
    estimate: f64,
    standard_error: f64,
    method: &str,
) -> Result<CausalEffectEstimateV1, CausalDriftDiagnosisErrorV1> {
    if !estimate.is_finite() || !standard_error.is_finite() || standard_error < 0.0 {
        return Err(CausalDriftDiagnosisErrorV1(
            "effect estimate and standard error must be finite; standard error non-negative".into(),
        ));
    }
    let radius = NORMAL_95_Z * standard_error;
    let lower_95 = estimate - radius;
    let upper_95 = estimate + radius;
    if !lower_95.is_finite() || !upper_95.is_finite() {
        return Err(CausalDriftDiagnosisErrorV1("effect interval overflow".into()));
    }
    Ok(CausalEffectEstimateV1 {
        estimate,
        standard_error,
        lower_95,
        upper_95,
        method: method.into(),
    })
}

fn estimate_matched_contrast(
    contrast: &MatchedDifferenceInDifferencesV1,
    policy: &CausalDriftPolicyV1,
) -> Result<(CausalEffectEstimateV1, bool), CausalDriftDiagnosisErrorV1> {
    let values = [
        contrast.treated_pre_mean,
        contrast.treated_post_mean,
        contrast.control_pre_mean,
        contrast.control_post_mean,
        contrast.treated_pre_variance,
        contrast.treated_post_variance,
        contrast.control_pre_variance,
        contrast.control_post_variance,
        contrast.maximum_absolute_covariate_smd,
        contrast.treated_pretrend_slope,
        contrast.control_pretrend_slope,
    ];
    if values.iter().any(|value| !value.is_finite())
        || [
            contrast.treated_pre_variance,
            contrast.treated_post_variance,
            contrast.control_pre_variance,
            contrast.control_post_variance,
        ]
        .iter()
        .any(|variance| *variance < 0.0)
    {
        return Err(CausalDriftDiagnosisErrorV1(
            "matched contrast contains non-finite values or negative variance".into(),
        ));
    }
    for (name, count) in [
        ("treated_pre_count", contrast.treated_pre_count),
        ("treated_post_count", contrast.treated_post_count),
        ("control_pre_count", contrast.control_pre_count),
        ("control_post_count", contrast.control_post_count),
    ] {
        if count < policy.minimum_group_count {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "matched contrast {name} is below the minimum group count"
            )));
        }
    }
    if !valid_sha256(&contrast.protocol_sha256)
        || !valid_sha256(&contrast.source_summary_sha256)
    {
        return Err(CausalDriftDiagnosisErrorV1(
            "matched contrast protocol/source digest is malformed".into(),
        ));
    }

    let treated_delta = contrast.treated_post_mean - contrast.treated_pre_mean;
    let control_delta = contrast.control_post_mean - contrast.control_pre_mean;
    let estimate = treated_delta - control_delta;
    let variance = contrast.treated_pre_variance / contrast.treated_pre_count as f64
        + contrast.treated_post_variance / contrast.treated_post_count as f64
        + contrast.control_pre_variance / contrast.control_pre_count as f64
        + contrast.control_post_variance / contrast.control_post_count as f64;
    let standard_error = variance.sqrt();
    let effect = estimate_interval(
        estimate,
        standard_error,
        "difference_in_differences_independent_cell_wald_95",
    )?;

    let balance_ok = contrast.maximum_absolute_covariate_smd <= policy.maximum_covariate_smd;
    let pretrend_ok = (contrast.treated_pretrend_slope - contrast.control_pretrend_slope).abs()
        <= policy.maximum_pretrend_slope_difference;
    Ok((effect, balance_ok && pretrend_ok))
}

fn estimate_clustered_contrast(
    contrast: &ClusteredDifferenceInDifferencesV1,
    policy: &CausalDriftPolicyV1,
) -> Result<(CausalEffectEstimateV1, bool), CausalDriftDiagnosisErrorV1> {
    if contrast.clusters.len() < policy.minimum_independent_clusters {
        return Err(CausalDriftDiagnosisErrorV1(format!(
            "clustered contrast has {} independent clusters; at least {} are required",
            contrast.clusters.len(),
            policy.minimum_independent_clusters
        )));
    }
    if !valid_sha256(&contrast.protocol_sha256)
        || !valid_sha256(&contrast.source_summary_sha256)
    {
        return Err(CausalDriftDiagnosisErrorV1(
            "clustered contrast protocol/source digest is malformed".into(),
        ));
    }

    let mut cluster_ids = BTreeSet::new();
    let mut effects = Vec::with_capacity(contrast.clusters.len());
    let mut diagnostics_ok = true;
    for cluster in &contrast.clusters {
        require_text(&cluster.cluster_id, "clustered_contrast.cluster_id")?;
        if !cluster_ids.insert(cluster.cluster_id.as_str()) {
            return Err(CausalDriftDiagnosisErrorV1(
                "clustered contrast repeats an independent cluster ID".into(),
            ));
        }
        let values = [
            cluster.treated_pre_mean,
            cluster.treated_post_mean,
            cluster.control_pre_mean,
            cluster.control_post_mean,
            cluster.maximum_absolute_covariate_smd,
            cluster.treated_pretrend_slope,
            cluster.control_pretrend_slope,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "cluster {} contains non-finite summary values",
                cluster.cluster_id
            )));
        }
        for (name, count) in [
            ("treated_pre_count", cluster.treated_pre_count),
            ("treated_post_count", cluster.treated_post_count),
            ("control_pre_count", cluster.control_pre_count),
            ("control_post_count", cluster.control_post_count),
        ] {
            if count < policy.minimum_group_count {
                return Err(CausalDriftDiagnosisErrorV1(format!(
                    "cluster {} {name} is below the minimum group count",
                    cluster.cluster_id
                )));
            }
        }
        if cluster.maximum_absolute_covariate_smd > policy.maximum_covariate_smd
            || (cluster.treated_pretrend_slope - cluster.control_pretrend_slope).abs()
                > policy.maximum_pretrend_slope_difference
        {
            diagnostics_ok = false;
        }
        let treated_delta = cluster.treated_post_mean - cluster.treated_pre_mean;
        let control_delta = cluster.control_post_mean - cluster.control_pre_mean;
        let effect = treated_delta - control_delta;
        if !effect.is_finite() {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "cluster {} DiD effect is non-finite",
                cluster.cluster_id
            )));
        }
        effects.push(effect);
    }

    // Equal-weight cluster-average estimand. Standard error is computed from
    // between-cluster variation, not by treating wafer rows as independent.
    let cluster_count = effects.len();
    let estimate = effects.iter().sum::<f64>() / cluster_count as f64;
    let sum_squared_deviation = effects
        .iter()
        .map(|effect| (effect - estimate).powi(2))
        .sum::<f64>();
    let sample_variance = sum_squared_deviation / (cluster_count - 1) as f64;
    let standard_error = (sample_variance / cluster_count as f64).sqrt();
    if !estimate.is_finite() || !standard_error.is_finite() {
        return Err(CausalDriftDiagnosisErrorV1(
            "clustered contrast estimate or standard error is non-finite".into(),
        ));
    }
    let radius = T_95_DF9 * standard_error;
    let effect = CausalEffectEstimateV1 {
        estimate,
        standard_error,
        lower_95: estimate - radius,
        upper_95: estimate + radius,
        method: format!(
            "equal_weight_cluster_mean_did_student_t95_conservative_df9_n{cluster_count}"
        ),
    };
    if !effect.lower_95.is_finite() || !effect.upper_95.is_finite() {
        return Err(CausalDriftDiagnosisErrorV1(
            "clustered contrast interval overflows".into(),
        ));
    }
    Ok((effect, diagnostics_ok))
}

fn estimate_intervention(
    intervention: &ControlledInterventionEvidenceV1,
    policy: &CausalDriftPolicyV1,
) -> Result<(CausalEffectEstimateV1, bool), CausalDriftDiagnosisErrorV1> {
    if intervention.treated_count < policy.minimum_group_count
        || intervention.control_count < policy.minimum_group_count
    {
        return Err(CausalDriftDiagnosisErrorV1(
            "controlled intervention has insufficient treated/control support".into(),
        ));
    }
    if !valid_sha256(&intervention.assignment_protocol_sha256)
        || !valid_sha256(&intervention.preregistration_sha256)
        || !valid_sha256(&intervention.independent_verification_receipt_sha256)
    {
        return Err(CausalDriftDiagnosisErrorV1(
            "controlled intervention protocol/verification digest is malformed".into(),
        ));
    }
    let effect = estimate_interval(
        intervention.effect_estimate,
        intervention.standard_error,
        "controlled_intervention_wald_95",
    )?;
    let design_ok = intervention.randomized_assignment
        && intervention.preregistered_analysis
        && intervention.control_group_present
        && intervention.manipulation_check_passed
        && intervention.interference_assumption_declared;
    Ok((effect, design_ok))
}

fn effect_consistent(
    effect: &CausalEffectEstimateV1,
    direction: ExpectedEffectDirectionV1,
) -> bool {
    match direction {
        ExpectedEffectDirectionV1::Positive => effect.lower_95 > 0.0,
        ExpectedEffectDirectionV1::Negative => effect.upper_95 < 0.0,
        ExpectedEffectDirectionV1::NonZero => effect.lower_95 > 0.0 || effect.upper_95 < 0.0,
    }
}

fn effect_contradicts(
    effect: &CausalEffectEstimateV1,
    direction: ExpectedEffectDirectionV1,
) -> bool {
    match direction {
        ExpectedEffectDirectionV1::Positive => effect.upper_95 < 0.0,
        ExpectedEffectDirectionV1::Negative => effect.lower_95 > 0.0,
        ExpectedEffectDirectionV1::NonZero => false,
    }
}

fn negative_control_violations(
    controls: &[NegativeControlEvidenceV1],
) -> Result<Vec<String>, CausalDriftDiagnosisErrorV1> {
    let mut violations = Vec::new();
    let mut seen = BTreeSet::new();
    for control in controls {
        require_text(&control.control_id, "negative_control.control_id")?;
        if !seen.insert(control.control_id.as_str()) {
            return Err(CausalDriftDiagnosisErrorV1(
                "duplicate negative-control identity".into(),
            ));
        }
        if !valid_sha256(&control.source_artifact_sha256)
            || !control.effect_estimate.is_finite()
            || !control.standard_error.is_finite()
            || control.standard_error < 0.0
            || !control.materiality_margin.is_finite()
            || control.materiality_margin < 0.0
        {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "negative control {} has invalid numeric evidence",
                control.control_id
            )));
        }
        let radius = NORMAL_95_Z * control.standard_error;
        let lower = control.effect_estimate - radius;
        let upper = control.effect_estimate + radius;
        if lower > control.materiality_margin || upper < -control.materiality_margin {
            violations.push(control.control_id.clone());
        }
    }
    violations.sort();
    Ok(violations)
}

fn validate_evidence(
    hypothesis: &CausalDriftHypothesisInputV1,
    input: &CausalDriftDiagnosisInputV1,
) -> Result<(Vec<String>, usize, bool), CausalDriftDiagnosisErrorV1> {
    let mut ids = BTreeSet::new();
    let mut evidence_ids = Vec::new();
    let mut observational_count = 0;
    let mut timing_ok = true;
    for evidence in &hypothesis.evidence {
        if evidence.schema != CAUSAL_DRIFT_EVIDENCE_SCHEMA_V1 {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "hypothesis {} has an unsupported evidence schema",
                hypothesis.hypothesis_id
            )));
        }
        require_text(&evidence.evidence_id, "evidence.evidence_id")?;
        require_text(&evidence.hypothesis_id, "evidence.hypothesis_id")?;
        require_text(&evidence.clock_domain_id, "evidence.clock_domain_id")?;
        require_text(&evidence.authority_ref, "evidence.authority_ref")?;
        if !valid_sha256(&evidence.source_artifact_sha256) {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "evidence {} source digest is malformed",
                evidence.evidence_id
            )));
        }
        if !ids.insert(evidence.evidence_id.as_str()) {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "hypothesis {} repeats evidence ID {}",
                hypothesis.hypothesis_id, evidence.evidence_id
            )));
        }
        if evidence.hypothesis_id != hypothesis.hypothesis_id {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "evidence {} is bound to another hypothesis",
                evidence.evidence_id
            )));
        }
        if evidence.clock_domain_id != input.incident_clock_domain_id
            || evidence.available_at_unix_seconds < evidence.event_time_unix_seconds
            || evidence.available_at_unix_seconds > input.analysis_cutoff_unix_seconds
        {
            timing_ok = false;
        }
        if matches!(
            evidence.evidence_kind,
            CausalDriftEvidenceKindV1::FeatureAttribution
                | CausalDriftEvidenceKindV1::TemporalPrecedence
                | CausalDriftEvidenceKindV1::MaintenanceRecord
                | CausalDriftEvidenceKindV1::ReferenceMeasurement
                | CausalDriftEvidenceKindV1::ClockJoinAudit
                | CausalDriftEvidenceKindV1::SourceInventory
        ) {
            observational_count += 1;
        }
        if evidence.attribution_score.is_some_and(|score| !score.is_finite()) {
            return Err(CausalDriftDiagnosisErrorV1(format!(
                "evidence {} has a non-finite attribution score",
                evidence.evidence_id
            )));
        }
        evidence_ids.push(evidence.evidence_id.clone());
    }
    evidence_ids.sort();
    Ok((evidence_ids, observational_count, timing_ok))
}

fn assess_hypothesis(
    hypothesis: &CausalDriftHypothesisInputV1,
    input: &CausalDriftDiagnosisInputV1,
) -> Result<CausalHypothesisAssessmentV1, CausalDriftDiagnosisErrorV1> {
    require_text(&hypothesis.hypothesis_id, "hypothesis.hypothesis_id")?;
    require_text(&hypothesis.scope, "hypothesis.scope")?;
    let (evidence_ids, observational_evidence_count, timing_ok) =
        validate_evidence(hypothesis, input)?;
    let mut reasons = Vec::new();
    let mut status = CausalAssessmentStatusV1::AssociationOnly;
    let mut matched_effect = None;
    let mut intervention_effect = None;

    if !input.source_inventory_complete || !input.clock_join_integrity || !timing_ok {
        status = CausalAssessmentStatusV1::Unknown;
        if !input.source_inventory_complete {
            reasons.push("source_inventory_incomplete".into());
        }
        if !input.clock_join_integrity {
            reasons.push("clock_join_integrity_unqualified".into());
        }
        if !timing_ok {
            reasons.push("evidence_clock_or_availability_mismatch".into());
        }
    }

    let has_matched_evidence = hypothesis
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_kind == CausalDriftEvidenceKindV1::MatchedContrast);
    let has_intervention_evidence = hypothesis
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_kind == CausalDriftEvidenceKindV1::ControlledIntervention);
    if hypothesis.matched_contrast.is_some() && hypothesis.clustered_matched_contrast.is_some() {
        return Err(CausalDriftDiagnosisErrorV1(format!(
            "hypothesis {} cannot supply both cell-level and cluster-level matched contrasts",
            hypothesis.hypothesis_id
        )));
    }
    if (hypothesis.matched_contrast.is_some() || hypothesis.clustered_matched_contrast.is_some())
        && !has_matched_evidence
    {
        return Err(CausalDriftDiagnosisErrorV1(format!(
            "hypothesis {} has a matched contrast without a typed evidence record",
            hypothesis.hypothesis_id
        )));
    }
    if hypothesis.controlled_intervention.is_some() && !has_intervention_evidence {
        return Err(CausalDriftDiagnosisErrorV1(format!(
            "hypothesis {} has intervention statistics without a typed evidence record",
            hypothesis.hypothesis_id
        )));
    }

    let violations = negative_control_violations(&hypothesis.negative_controls)?;
    if status != CausalAssessmentStatusV1::Unknown && !violations.is_empty() {
        status = CausalAssessmentStatusV1::Unidentifiable;
        reasons.push("negative_control_effect_exceeds_materiality_margin".into());
    }

    if status == CausalAssessmentStatusV1::AssociationOnly {
        if let Some(contrast) = &hypothesis.matched_contrast {
            let (effect, diagnostics_ok) = estimate_matched_contrast(contrast, &input.policy)?;
            matched_effect = Some(effect.clone());
            if !diagnostics_ok {
                status = CausalAssessmentStatusV1::Unidentifiable;
                reasons.push("matched_balance_or_parallel_trend_gate_failed".into());
            } else if effect_contradicts(&effect, hypothesis.expected_effect_direction) {
                status = CausalAssessmentStatusV1::Contradicted;
                reasons.push("matched_contrast_confidence_interval_opposes_hypothesis".into());
            } else {
                // Even a well-balanced DiD result is observational and cannot
                // promote a physical-cause claim in the presence of hidden confounding.
                status = CausalAssessmentStatusV1::AssociationOnly;
                reasons.push("matched_contrast_is_association_not_causal_proof".into());
                if !effect_consistent(&effect, hypothesis.expected_effect_direction) {
                    reasons.push("matched_effect_not_separated_from_zero".into());
                }
                if !hypothesis.unmeasured_confounders.is_empty() {
                    reasons.push("unmeasured_confounders_remain".into());
                }
            }
        } else if let Some(contrast) = &hypothesis.clustered_matched_contrast {
            let (effect, diagnostics_ok) = estimate_clustered_contrast(contrast, &input.policy)?;
            matched_effect = Some(effect.clone());
            if !diagnostics_ok {
                status = CausalAssessmentStatusV1::Unidentifiable;
                reasons.push("clustered_balance_or_parallel_trend_gate_failed".into());
            } else if effect_contradicts(&effect, hypothesis.expected_effect_direction) {
                status = CausalAssessmentStatusV1::Contradicted;
                reasons.push("clustered_contrast_interval_opposes_hypothesis".into());
            } else {
                status = CausalAssessmentStatusV1::AssociationOnly;
                reasons.push("clustered_contrast_is_association_not_causal_proof".into());
                if !effect_consistent(&effect, hypothesis.expected_effect_direction) {
                    reasons.push("clustered_effect_not_separated_from_zero".into());
                }
                if !hypothesis.unmeasured_confounders.is_empty() {
                    reasons.push("unmeasured_confounders_remain".into());
                }
            }
        }
    }

    if status == CausalAssessmentStatusV1::AssociationOnly {
        if let Some(intervention) = &hypothesis.controlled_intervention {
            let (effect, design_ok) = estimate_intervention(intervention, &input.policy)?;
            intervention_effect = Some(effect.clone());
            if !design_ok {
                status = CausalAssessmentStatusV1::Unidentifiable;
                reasons.push("controlled_intervention_design_gate_failed".into());
            } else if !violations.is_empty() {
                status = CausalAssessmentStatusV1::Unidentifiable;
            } else if effect_contradicts(&effect, hypothesis.expected_effect_direction) {
                status = CausalAssessmentStatusV1::Contradicted;
                reasons.push("controlled_intervention_effect_opposes_hypothesis".into());
            } else if effect_consistent(&effect, hypothesis.expected_effect_direction) {
                status = CausalAssessmentStatusV1::InterventionConsistent;
                reasons.push("randomized_preregistered_intervention_is_consistent".into());
                reasons.push("independent_human_root_cause_review_still_required".into());
            } else {
                status = CausalAssessmentStatusV1::AssociationOnly;
                reasons.push("controlled_intervention_effect_not_separated_from_zero".into());
            }
        }
    }

    if evidence_ids.is_empty() && status == CausalAssessmentStatusV1::AssociationOnly {
        status = CausalAssessmentStatusV1::Unknown;
        reasons.push("no_evidence_records_supplied".into());
    }
    if hypothesis.hypothesis == CausalDriftHypothesisKindV1::UnknownOrUnmodelled
        && status == CausalAssessmentStatusV1::InterventionConsistent
    {
        status = CausalAssessmentStatusV1::Unidentifiable;
        reasons.push("unknown_hypothesis_vocabulary_cannot_be_promoted".into());
    }
    reasons.sort();
    reasons.dedup();

    let mut sorted_evidence_ids = evidence_ids;
    sorted_evidence_ids.sort();
    Ok(CausalHypothesisAssessmentV1 {
        hypothesis_id: hypothesis.hypothesis_id.clone(),
        hypothesis: hypothesis.hypothesis,
        scope: hypothesis.scope.clone(),
        status,
        reasons,
        evidence_ids: sorted_evidence_ids,
        observational_evidence_count,
        matched_contrast: matched_effect,
        controlled_intervention: intervention_effect,
        negative_control_violation_ids: violations,
        measured_confounders: {
            let mut values = hypothesis.measured_confounders.clone();
            values.sort();
            values.dedup();
            values
        },
        unmeasured_confounders: {
            let mut values = hypothesis.unmeasured_confounders.clone();
            values.sort();
            values.dedup();
            values
        },
        causal_claim_authorized: false,
        human_review_required: true,
    })
}

/// Seals and evaluates a set of competing drift hypotheses. Any output is a
/// diagnostic hypothesis assessment; causal claims and physical actions remain
/// unauthorized by this software layer.
pub fn diagnose_causal_drift_v1(
    input: &CausalDriftDiagnosisInputV1,
) -> Result<CausalDriftDiagnosisReportV1, CausalDriftDiagnosisErrorV1> {
    if input.schema != CAUSAL_DRIFT_INPUT_SCHEMA_V1 {
        return Err(CausalDriftDiagnosisErrorV1(
            "causal drift input schema is unsupported".into(),
        ));
    }
    require_text(&input.incident_id, "incident_id")?;
    require_text(&input.incident_clock_domain_id, "incident_clock_domain_id")?;
    if input.analysis_cutoff_unix_seconds < input.incident_time_unix_seconds {
        return Err(CausalDriftDiagnosisErrorV1(
            "analysis cutoff cannot precede the incident time".into(),
        ));
    }
    if !valid_sha256(&input.source_inventory_sha256)
        || !valid_sha256(&input.clock_join_audit_sha256)
    {
        return Err(CausalDriftDiagnosisErrorV1(
            "source inventory or clock-join audit digest is malformed".into(),
        ));
    }
    if input.policy.minimum_group_count < 2
        || input.policy.minimum_independent_clusters < 10
        || !input.policy.maximum_covariate_smd.is_finite()
        || input.policy.maximum_covariate_smd < 0.0
        || !input.policy.maximum_pretrend_slope_difference.is_finite()
        || input.policy.maximum_pretrend_slope_difference < 0.0
    {
        return Err(CausalDriftDiagnosisErrorV1(
            "causal diagnosis policy contains invalid thresholds or fewer than 10 independent clusters".into(),
        ));
    }
    if input.hypotheses.is_empty() {
        return Err(CausalDriftDiagnosisErrorV1(
            "at least one causal hypothesis is required".into(),
        ));
    }
    let mut hypothesis_ids = BTreeSet::new();
    let mut assessments = Vec::with_capacity(input.hypotheses.len());
    for hypothesis in &input.hypotheses {
        if !hypothesis_ids.insert(hypothesis.hypothesis_id.as_str()) {
            return Err(CausalDriftDiagnosisErrorV1(
                "duplicate causal hypothesis ID".into(),
            ));
        }
        assessments.push(assess_hypothesis(hypothesis, input)?);
    }
    assessments.sort_by(|left, right| left.hypothesis_id.cmp(&right.hypothesis_id));

    let mut report = CausalDriftDiagnosisReportV1 {
        schema: CAUSAL_DRIFT_REPORT_SCHEMA_V1.into(),
        incident_id: input.incident_id.clone(),
        analysis_cutoff_unix_seconds: input.analysis_cutoff_unix_seconds,
        input_sha256: input_digest(input)?,
        status_by_hypothesis: assessments,
        source_authenticity_proven: false,
        causal_claim_authorized: false,
        physical_action_authorized: false,
        equipment_control_authority: false,
        claim_ceiling: vec![
            "Feature attribution and temporal precedence are association evidence only".into(),
            "Matched difference-in-differences cannot eliminate unmeasured confounding".into(),
            "Intervention consistency requires independent physical and human review".into(),
            "Source authenticity is not established by content digests".into(),
            "No physical action or equipment control is authorized".into(),
        ],
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn evidence(
        hypothesis_id: &str,
        evidence_id: &str,
        kind: CausalDriftEvidenceKindV1,
    ) -> CausalDriftEvidenceV1 {
        CausalDriftEvidenceV1 {
            schema: CAUSAL_DRIFT_EVIDENCE_SCHEMA_V1.into(),
            evidence_id: evidence_id.into(),
            evidence_kind: kind,
            hypothesis_id: hypothesis_id.into(),
            event_time_unix_seconds: 100,
            available_at_unix_seconds: 110,
            clock_domain_id: "fab-clock".into(),
            source_artifact_sha256: digest('a'),
            authority_ref: "receipt://evidence/1".into(),
            attribution_score: Some(0.99),
        }
    }

    fn contrast() -> MatchedDifferenceInDifferencesV1 {
        MatchedDifferenceInDifferencesV1 {
            treated_pre_mean: 10.0,
            treated_post_mean: 14.0,
            control_pre_mean: 10.0,
            control_post_mean: 11.0,
            treated_pre_variance: 1.0,
            treated_post_variance: 1.0,
            control_pre_variance: 1.0,
            control_post_variance: 1.0,
            treated_pre_count: 100,
            treated_post_count: 100,
            control_pre_count: 100,
            control_post_count: 100,
            maximum_absolute_covariate_smd: 0.05,
            treated_pretrend_slope: 0.10,
            control_pretrend_slope: 0.11,
            protocol_sha256: digest('b'),
            source_summary_sha256: digest('c'),
        }
    }

    fn clustered_contrast_with_count(count: usize) -> ClusteredDifferenceInDifferencesV1 {
        let clusters = (1..=count)
            .map(|effect| ClusterDidSummaryV1 {
                cluster_id: format!("chamber-{effect}"),
                treated_pre_mean: 10.0,
                treated_post_mean: 11.0 + effect as f64,
                control_pre_mean: 10.0,
                control_post_mean: 11.0,
                treated_pre_count: 20,
                treated_post_count: 20,
                control_pre_count: 20,
                control_post_count: 20,
                maximum_absolute_covariate_smd: 0.05,
                treated_pretrend_slope: 0.10,
                control_pretrend_slope: 0.11,
            })
            .collect();
        ClusteredDifferenceInDifferencesV1 {
            clusters,
            protocol_sha256: digest('b'),
            source_summary_sha256: digest('c'),
        }
    }

    fn clustered_contrast() -> ClusteredDifferenceInDifferencesV1 {
        clustered_contrast_with_count(10)
    }

    fn hypothesis(
        id: &str,
        kind: CausalDriftHypothesisKindV1,
        evidence: Vec<CausalDriftEvidenceV1>,
    ) -> CausalDriftHypothesisInputV1 {
        CausalDriftHypothesisInputV1 {
            hypothesis_id: id.into(),
            hypothesis: kind,
            expected_effect_direction: ExpectedEffectDirectionV1::Positive,
            scope: "tool-1/chamber-2".into(),
            evidence,
            matched_contrast: None,
            clustered_matched_contrast: None,
            controlled_intervention: None,
            negative_controls: Vec::new(),
            measured_confounders: vec!["recipe_id".into()],
            unmeasured_confounders: vec!["unknown_resist_lot".into()],
        }
    }

    fn input(hypotheses: Vec<CausalDriftHypothesisInputV1>) -> CausalDriftDiagnosisInputV1 {
        CausalDriftDiagnosisInputV1 {
            schema: CAUSAL_DRIFT_INPUT_SCHEMA_V1.into(),
            incident_id: "incident-001".into(),
            incident_clock_domain_id: "fab-clock".into(),
            incident_time_unix_seconds: 100,
            analysis_cutoff_unix_seconds: 200,
            source_inventory_complete: true,
            source_inventory_sha256: digest('d'),
            clock_join_integrity: true,
            clock_join_audit_sha256: digest('e'),
            policy: CausalDriftPolicyV1 {
                minimum_group_count: 10,
                maximum_covariate_smd: 0.10,
                maximum_pretrend_slope_difference: 0.05,
                minimum_independent_clusters: 10,
            },
            hypotheses,
        }
    }

    fn intervention() -> ControlledInterventionEvidenceV1 {
        ControlledInterventionEvidenceV1 {
            randomized_assignment: true,
            preregistered_analysis: true,
            control_group_present: true,
            manipulation_check_passed: true,
            interference_assumption_declared: true,
            treated_count: 100,
            control_count: 100,
            effect_estimate: 2.0,
            standard_error: 0.25,
            assignment_protocol_sha256: digest('f'),
            preregistration_sha256: digest('1'),
            independent_verification_receipt_sha256: digest('2'),
        }
    }

    #[test]
    fn feature_attribution_never_becomes_a_causal_claim() {
        let candidate = hypothesis(
            "sensor",
            CausalDriftHypothesisKindV1::SensorDrift,
            vec![evidence(
                "sensor",
                "shap-feature-current",
                CausalDriftEvidenceKindV1::FeatureAttribution,
            )],
        );
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        let assessment = &report.status_by_hypothesis[0];
        assert_eq!(assessment.status, CausalAssessmentStatusV1::AssociationOnly);
        assert!(!assessment.causal_claim_authorized);
        assert!(!report.causal_claim_authorized);
    }

    #[test]
    fn matched_difference_in_differences_is_association_only_even_when_effect_is_large() {
        let mut candidate = hypothesis(
            "process",
            CausalDriftHypothesisKindV1::RecipeOrProcessChange,
            vec![evidence("process", "matched", CausalDriftEvidenceKindV1::MatchedContrast)],
        );
        candidate.matched_contrast = Some(contrast());
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        let assessment = &report.status_by_hypothesis[0];
        assert_eq!(assessment.status, CausalAssessmentStatusV1::AssociationOnly);
        let estimate = assessment.matched_contrast.as_ref().unwrap();
        assert!((estimate.estimate - 3.0).abs() < 1.0e-12);
        assert!(!assessment.causal_claim_authorized);
    }

    #[test]
    fn clustered_did_uses_between_cluster_variance_and_stays_association_only() {
        let mut candidate = hypothesis(
            "process",
            CausalDriftHypothesisKindV1::RecipeOrProcessChange,
            vec![evidence(
                "process",
                "clustered-matched",
                CausalDriftEvidenceKindV1::MatchedContrast,
            )],
        );
        candidate.clustered_matched_contrast = Some(clustered_contrast());
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        let assessment = &report.status_by_hypothesis[0];
        assert_eq!(assessment.status, CausalAssessmentStatusV1::AssociationOnly);
        let estimate = assessment.matched_contrast.as_ref().unwrap();
        assert!((estimate.estimate - 5.5).abs() < 1.0e-12);
        assert!((estimate.standard_error - (11.0_f64 / 12.0).sqrt()).abs() < 1.0e-12);
        assert!(estimate.method.contains("cluster_mean"));
        assert!(!assessment.causal_claim_authorized);
    }

    #[test]
    fn clustered_did_rejects_insufficient_independent_clusters() {
        let mut candidate = hypothesis(
            "process",
            CausalDriftHypothesisKindV1::RecipeOrProcessChange,
            vec![evidence(
                "process",
                "clustered-matched",
                CausalDriftEvidenceKindV1::MatchedContrast,
            )],
        );
        candidate.clustered_matched_contrast = Some(clustered_contrast_with_count(9));
        assert!(diagnose_causal_drift_v1(&input(vec![candidate])).is_err());
    }

    #[test]
    fn cell_and_cluster_matched_contrasts_cannot_be_mixed() {
        let mut candidate = hypothesis(
            "process",
            CausalDriftHypothesisKindV1::RecipeOrProcessChange,
            vec![evidence(
                "process",
                "matched",
                CausalDriftEvidenceKindV1::MatchedContrast,
            )],
        );
        candidate.matched_contrast = Some(contrast());
        candidate.clustered_matched_contrast = Some(clustered_contrast());
        assert!(diagnose_causal_drift_v1(&input(vec![candidate])).is_err());
    }

    #[test]
    fn poor_balance_or_pretrend_downgrades_matched_contrast() {
        let mut candidate = hypothesis(
            "process",
            CausalDriftHypothesisKindV1::RecipeOrProcessChange,
            vec![evidence("process", "matched", CausalDriftEvidenceKindV1::MatchedContrast)],
        );
        let mut poor = contrast();
        poor.maximum_absolute_covariate_smd = 0.40;
        candidate.matched_contrast = Some(poor);
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        assert_eq!(
            report.status_by_hypothesis[0].status,
            CausalAssessmentStatusV1::Unidentifiable
        );
    }

    #[test]
    fn negative_control_violation_blocks_causal_promotion() {
        let mut candidate = hypothesis(
            "process",
            CausalDriftHypothesisKindV1::RecipeOrProcessChange,
            vec![evidence("process", "matched", CausalDriftEvidenceKindV1::MatchedContrast)],
        );
        candidate.matched_contrast = Some(contrast());
        candidate.negative_controls.push(NegativeControlEvidenceV1 {
            control_id: "unaffected-measurand".into(),
            source_artifact_sha256: digest('a'),
            effect_estimate: 2.0,
            standard_error: 0.1,
            materiality_margin: 0.2,
        });
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        let assessment = &report.status_by_hypothesis[0];
        assert_eq!(assessment.status, CausalAssessmentStatusV1::Unidentifiable);
        assert_eq!(
            assessment.negative_control_violation_ids,
            vec!["unaffected-measurand".to_string()]
        );
    }

    #[test]
    fn randomized_preregistered_intervention_can_be_consistent_but_never_authorizes_a_claim() {
        let mut candidate = hypothesis(
            "repair",
            CausalDriftHypothesisKindV1::ToolStateChange,
            vec![evidence("repair", "trial", CausalDriftEvidenceKindV1::ControlledIntervention)],
        );
        candidate.controlled_intervention = Some(intervention());
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        let assessment = &report.status_by_hypothesis[0];
        assert_eq!(assessment.status, CausalAssessmentStatusV1::InterventionConsistent);
        assert!(assessment.human_review_required);
        assert!(!assessment.causal_claim_authorized);
        assert!(!report.physical_action_authorized);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn intervention_statistics_without_intervention_evidence_are_rejected() {
        let mut candidate = hypothesis(
            "repair",
            CausalDriftHypothesisKindV1::ToolStateChange,
            vec![evidence(
                "repair",
                "only-attribution",
                CausalDriftEvidenceKindV1::FeatureAttribution,
            )],
        );
        candidate.controlled_intervention = Some(intervention());
        assert!(diagnose_causal_drift_v1(&input(vec![candidate])).is_err());
    }

    #[test]
    fn non_randomized_intervention_is_unidentifiable() {
        let mut candidate = hypothesis(
            "repair",
            CausalDriftHypothesisKindV1::ToolStateChange,
            vec![evidence("repair", "trial", CausalDriftEvidenceKindV1::ControlledIntervention)],
        );
        let mut trial = intervention();
        trial.randomized_assignment = false;
        candidate.controlled_intervention = Some(trial);
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        assert_eq!(
            report.status_by_hypothesis[0].status,
            CausalAssessmentStatusV1::Unidentifiable
        );
    }

    #[test]
    fn incomplete_inventory_and_clock_join_return_unknown() {
        let candidate = hypothesis(
            "sensor",
            CausalDriftHypothesisKindV1::SensorDrift,
            vec![evidence("sensor", "observed", CausalDriftEvidenceKindV1::TemporalPrecedence)],
        );
        let mut incomplete = input(vec![candidate.clone()]);
        incomplete.source_inventory_complete = false;
        let report = diagnose_causal_drift_v1(&incomplete).unwrap();
        assert_eq!(report.status_by_hypothesis[0].status, CausalAssessmentStatusV1::Unknown);

        let mut unaligned = input(vec![candidate]);
        unaligned.clock_join_integrity = false;
        let report = diagnose_causal_drift_v1(&unaligned).unwrap();
        assert_eq!(report.status_by_hypothesis[0].status, CausalAssessmentStatusV1::Unknown);
    }

    #[test]
    fn evidence_after_analysis_cutoff_or_from_another_clock_is_unknown() {
        let mut candidate = hypothesis(
            "sensor",
            CausalDriftHypothesisKindV1::SensorDrift,
            vec![evidence("sensor", "observed", CausalDriftEvidenceKindV1::TemporalPrecedence)],
        );
        candidate.evidence[0].available_at_unix_seconds = 201;
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.status_by_hypothesis[0].status, CausalAssessmentStatusV1::Unknown);

        let mut candidate = hypothesis(
            "sensor",
            CausalDriftHypothesisKindV1::SensorDrift,
            vec![evidence("sensor", "observed", CausalDriftEvidenceKindV1::TemporalPrecedence)],
        );
        candidate.evidence[0].clock_domain_id = "other-clock".into();
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.status_by_hypothesis[0].status, CausalAssessmentStatusV1::Unknown);
    }

    #[test]
    fn contradicted_effect_direction_is_retained_as_negative_evidence() {
        let mut candidate = hypothesis(
            "sensor",
            CausalDriftHypothesisKindV1::SensorDrift,
            vec![evidence("sensor", "matched", CausalDriftEvidenceKindV1::MatchedContrast)],
        );
        let mut opposite = contrast();
        opposite.treated_post_mean = 7.0;
        candidate.matched_contrast = Some(opposite);
        let report = diagnose_causal_drift_v1(&input(vec![candidate])).unwrap();
        assert_eq!(
            report.status_by_hypothesis[0].status,
            CausalAssessmentStatusV1::Contradicted
        );
    }

    #[test]
    fn input_order_does_not_change_sealed_digest_or_assessment_order() {
        let first_hypothesis = hypothesis(
            "b-sensor",
            CausalDriftHypothesisKindV1::SensorDrift,
            vec![evidence("b-sensor", "ev-b", CausalDriftEvidenceKindV1::FeatureAttribution)],
        );
        let second_hypothesis = hypothesis(
            "a-process",
            CausalDriftHypothesisKindV1::RecipeOrProcessChange,
            vec![evidence("a-process", "ev-a", CausalDriftEvidenceKindV1::TemporalPrecedence)],
        );
        let first = input(vec![first_hypothesis.clone(), second_hypothesis.clone()]);
        let digest_first = seal_causal_drift_diagnosis_input_v1(&first).unwrap();
        let mut second = input(vec![second_hypothesis, first_hypothesis]);
        second.hypotheses.reverse();
        assert_eq!(
            digest_first,
            seal_causal_drift_diagnosis_input_v1(&second).unwrap()
        );
        let report = diagnose_causal_drift_v1(&first).unwrap();
        assert_eq!(
            report
                .status_by_hypothesis
                .iter()
                .map(|value| value.hypothesis_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a-process", "b-sensor"]
        );
    }

    #[test]
    fn malformed_digests_and_duplicate_hypothesis_ids_are_rejected() {
        let candidate = hypothesis(
            "sensor",
            CausalDriftHypothesisKindV1::SensorDrift,
            vec![evidence("sensor", "observed", CausalDriftEvidenceKindV1::TemporalPrecedence)],
        );
        let mut malformed = input(vec![candidate.clone()]);
        malformed.source_inventory_sha256 = "not-a-sha".into();
        assert!(diagnose_causal_drift_v1(&malformed).is_err());

        let duplicate = input(vec![candidate.clone(), candidate]);
        assert!(diagnose_causal_drift_v1(&duplicate).is_err());
    }
}
