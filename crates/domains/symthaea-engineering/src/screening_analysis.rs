// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed descriptive analysis of measured responses from a preregistered factorial plan.
//!
//! The module computes factorial contrast estimates and per-block center-point diagnostics.
//! It does not calculate p-values, confidence intervals, exact standard errors, causal effects,
//! safety clearance, product release, or agronomic efficacy. A separate, preregistered statistical
//! model is required for inferential claims.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use symthaea_agribot::soil_process::{EvidenceKind, EvidenceRef};

use crate::screening_design::{
    verify_screening_design, PlannedRunKind, ScreeningDesignPlan,
    ScreeningDesignVerificationReceipt,
};

const RESPONSE_ANALYSIS_ALGORITHM_ID: &str = "blocked-full-factorial-descriptive-contrasts-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreeningAnalysisError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl ScreeningAnalysisError {
    fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl std::fmt::Display for ScreeningAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl std::error::Error for ScreeningAnalysisError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedFactorSetting {
    pub factor_id: String,
    pub value: f64,
    pub unit: String,
    /// Evidence for the achieved physical setting, not the assigned setpoint.
    pub evidence: EvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningResponseObservation {
    pub observation_id: String,
    pub run_id: String,
    /// Independently measured values actually achieved for each assigned process factor.
    pub actual_factor_settings: Vec<ObservedFactorSetting>,
    pub endpoint_id: String,
    pub outcome_value: f64,
    pub outcome_unit: String,
    pub measurement_method_id: String,
    /// Every analyzed outcome must reference a measured physical/laboratory result.
    pub evidence: EvidenceRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactorialEffectOrder {
    MainEffect,
    TwoFactorInteraction,
    HigherOrderInteraction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorialEffectEstimate {
    /// Deterministic ID based on original factor order, e.g. effect-01 or effect-01-02.
    pub effect_id: String,
    pub factor_ids: Vec<String>,
    pub order: u32,
    pub kind: FactorialEffectOrder,
    /// Difference of high vs low means for main effects; conventional factorial contrast
    /// effect for interactions. Same physical unit as the primary endpoint.
    pub effect_estimate: f64,
    pub unit: String,
    pub independent_complete_factorial_replicates: u32,
    /// Observation records used by this descriptive contrast.
    pub observation_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockCenterPointDiagnostic {
    pub block_id: String,
    pub factorial_treatment_mean: f64,
    pub center_point_mean: f64,
    /// Center-point mean minus the mean of the factorial treatment runs in the same block.
    /// This descriptive difference is not a formal curvature test or significance claim.
    pub center_point_minus_factorial_mean: f64,
    pub unit: String,
    pub center_point_observation_count: u32,
    pub factorial_treatment_observation_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningResponseAnalysis {
    pub algorithm_id: String,
    pub experiment_id: String,
    pub preregistration_id: String,
    pub input_snapshot_id: String,
    pub analysis_plan_id: String,
    pub endpoint_id: String,
    pub endpoint_unit: String,
    pub design_verification: ScreeningDesignVerificationReceipt,
    /// Full schedule request retained to keep the observed results interpretable.
    pub request_snapshot: crate::screening_design::ScreeningDesignRequest,
    /// Exact scheduled runs and settings analyzed; avoids relying on a future generator
    /// to reconstruct historical treatment allocation.
    pub schedule_snapshot: Vec<crate::screening_design::PlannedRun>,
    /// Ordered by planned run_order, preserving the audit trail for each observation.
    pub observations_in_run_order: Vec<ScreeningResponseObservation>,
    pub treatment_observation_count: u32,
    pub center_point_observation_count: u32,
    pub factorial_effects: Vec<FactorialEffectEstimate>,
    pub block_center_point_diagnostics: Vec<BlockCenterPointDiagnostic>,
    pub factorial_treatment_mean: f64,
    pub all_observation_count: u32,
    pub scope_note: String,
}

/// Analyze a complete, verified primary-endpoint dataset against the exact generated plan.
/// One measured observation is required for every planned run, including center-point runs.
/// Missing/duplicate outcomes are rejected instead of silently imputed. The output contains
/// descriptive factorial contrasts only; inferential analysis must follow the preregistered
/// analysis plan and account for the actual block/replicate structure.
pub fn analyze_screening_responses(
    plan: &ScreeningDesignPlan,
    observations: &[ScreeningResponseObservation],
) -> Result<ScreeningResponseAnalysis, ScreeningAnalysisError> {
    let design_verification = verify_screening_design(plan)
        .map_err(|error| ScreeningAnalysisError::new(error.field, error.reason))?;
    let request = &plan.request_snapshot;
    let endpoint = &request.primary_endpoint;
    let factors = &request.factors;
    let factor_count = factors.len();
    let combination_count = 1_usize << factor_count;

    if observations.len() != plan.runs.len() {
        return Err(ScreeningAnalysisError::new(
            "observations",
            "exactly one primary-endpoint observation is required for every planned run; missing outcomes need a separate declared missing-data analysis",
        ));
    }

    let planned_by_id: HashMap<&str, &crate::screening_design::PlannedRun> =
        plan.runs.iter().map(|run| (run.run_id.as_str(), run)).collect();
    let mut observation_ids = HashSet::new();
    let mut observations_by_run: HashMap<&str, &ScreeningResponseObservation> = HashMap::new();

    for observation in observations {
        if observation.observation_id.trim().is_empty() {
            return Err(ScreeningAnalysisError::new(
                "observations.observation_id",
                "cannot be empty",
            ));
        }
        if !observation_ids.insert(observation.observation_id.as_str()) {
            return Err(ScreeningAnalysisError::new(
                "observations.observation_id",
                "observation IDs must be unique",
            ));
        }
        let run = planned_by_id.get(observation.run_id.as_str()).ok_or_else(|| {
            ScreeningAnalysisError::new(
                "observations.run_id",
                "observation references a run that is not in the verified plan",
            )
        })?;
        if !observation.run_id.eq(run.run_id.as_str()) {
            return Err(ScreeningAnalysisError::new(
                "observations.run_id",
                "observation/run identity mismatch",
            ));
        }
        if observation.actual_factor_settings.len() != factor_count {
            return Err(ScreeningAnalysisError::new(
                "observations.actual_factor_settings",
                "one measured achieved setting is required for each preregistered factor",
            ));
        }
        let mut observed_factor_ids = HashSet::new();
        for actual in &observation.actual_factor_settings {
            if actual.factor_id.trim().is_empty()
                || !observed_factor_ids.insert(actual.factor_id.as_str())
            {
                return Err(ScreeningAnalysisError::new(
                    "observations.actual_factor_settings.factor_id",
                    "factor IDs must be non-empty and unique for each run",
                ));
            }
            let planned = run
                .settings
                .iter()
                .find(|setting| setting.factor_id == actual.factor_id)
                .ok_or_else(|| {
                    ScreeningAnalysisError::new(
                        "observations.actual_factor_settings.factor_id",
                        "observed factor is not present in the planned run",
                    )
                })?;
            let factor = factors
                .iter()
                .find(|factor| factor.factor_id == actual.factor_id)
                .ok_or_else(|| {
                    ScreeningAnalysisError::new(
                        "observations.actual_factor_settings.factor_id",
                        "observed factor is not present in the preregistered design",
                    )
                })?;
            if !actual.value.is_finite() {
                return Err(ScreeningAnalysisError::new(
                    "observations.actual_factor_settings.value",
                    "must be finite",
                ));
            }
            if actual.unit != factor.unit {
                return Err(ScreeningAnalysisError::new(
                    "observations.actual_factor_settings.unit",
                    "achieved setting unit must match the preregistered factor unit",
                ));
            }
            if actual.evidence.evidence_id.trim().is_empty()
                || actual.evidence.kind != EvidenceKind::Measured
            {
                return Err(ScreeningAnalysisError::new(
                    "observations.actual_factor_settings.evidence",
                    "achieved physical settings require evidence classified as measured",
                ));
            }
            let deviation = (actual.value - planned.value).abs();
            if !deviation.is_finite() || deviation > factor.max_absolute_deviation {
                return Err(ScreeningAnalysisError::new(
                    "observations.actual_factor_settings.value",
                    "achieved setting falls outside the preregistered absolute-deviation tolerance; use a separately reviewed deviation/missing-data analysis",
                ));
            }
        }
        if observed_factor_ids.len() != factor_count {
            return Err(ScreeningAnalysisError::new(
                "observations.actual_factor_settings",
                "the run does not contain every preregistered factor",
            ));
        }

        if observation.endpoint_id != endpoint.endpoint_id {
            return Err(ScreeningAnalysisError::new(
                "observations.endpoint_id",
                "observed endpoint must match the preregistered primary endpoint",
            ));
        }
        if observation.outcome_unit != endpoint.unit {
            return Err(ScreeningAnalysisError::new(
                "observations.outcome_unit",
                "outcome unit must exactly match the preregistered primary endpoint unit",
            ));
        }
        if observation.measurement_method_id != endpoint.measurement_method_id {
            return Err(ScreeningAnalysisError::new(
                "observations.measurement_method_id",
                "measurement method must match the preregistered primary endpoint method",
            ));
        }
        if !observation.outcome_value.is_finite() {
            return Err(ScreeningAnalysisError::new(
                "observations.outcome_value",
                "must be finite",
            ));
        }
        if observation.evidence.evidence_id.trim().is_empty()
            || observation.evidence.kind != EvidenceKind::Measured
        {
            return Err(ScreeningAnalysisError::new(
                "observations.evidence",
                "each outcome must have a non-empty evidence ID classified as measured",
            ));
        }
        if observations_by_run
            .insert(observation.run_id.as_str(), observation)
            .is_some()
        {
            return Err(ScreeningAnalysisError::new(
                "observations.run_id",
                "exactly one primary-endpoint outcome is allowed per planned run",
            ));
        }
    }

    if observations_by_run.len() != planned_by_id.len() {
        return Err(ScreeningAnalysisError::new(
            "observations",
            "one or more planned runs have no observed primary-endpoint result",
        ));
    }

    let mut ordered_observations: Vec<ScreeningResponseObservation> =
        Vec::with_capacity(plan.runs.len());
    for run in &plan.runs {
        let observation = observations_by_run
            .get(run.run_id.as_str())
            .copied()
            .ok_or_else(|| {
                ScreeningAnalysisError::new(
                    "observations",
                    "verified plan run has no primary-endpoint observation",
                )
            })?;
        ordered_observations.push((*observation).clone());
    }

    let mut cell_sum: HashMap<(String, u32), f64> = HashMap::new();
    let mut cell_count: HashMap<(String, u32), u32> = HashMap::new();
    let mut treatment_sum_by_block: HashMap<String, f64> = HashMap::new();
    let mut treatment_count_by_block: HashMap<String, u32> = HashMap::new();
    let mut center_sum_by_block: HashMap<String, f64> = HashMap::new();
    let mut center_count_by_block: HashMap<String, u32> = HashMap::new();
    let mut factorial_total = 0.0_f64;
    let mut treatment_observation_count = 0_u32;
    let mut center_point_observation_count = 0_u32;

    for run in &plan.runs {
        let outcome = observations_by_run[run.run_id.as_str()].outcome_value;
        match run.kind {
            PlannedRunKind::FactorialTreatment => {
                let standard_order = run.standard_order.ok_or_else(|| {
                    ScreeningAnalysisError::new("runs.standard_order", "treatment row is missing")
                })?;
                let key = (run.block_id.clone(), standard_order);
                let sum = cell_sum.entry(key.clone()).or_insert(0.0);
                *sum += outcome;
                if !sum.is_finite() {
                    return Err(ScreeningAnalysisError::new(
                        "observations.outcome_value",
                        "treatment-cell total overflow",
                    ));
                }
                *cell_count.entry(key).or_insert(0) += 1;

                let block_sum = treatment_sum_by_block
                    .entry(run.block_id.clone())
                    .or_insert(0.0);
                *block_sum += outcome;
                if !block_sum.is_finite() {
                    return Err(ScreeningAnalysisError::new(
                        "observations.outcome_value",
                        "block treatment total overflow",
                    ));
                }
                *treatment_count_by_block
                    .entry(run.block_id.clone())
                    .or_insert(0) += 1;
                factorial_total += outcome;
                if !factorial_total.is_finite() {
                    return Err(ScreeningAnalysisError::new(
                        "observations.outcome_value",
                        "factorial grand total overflow",
                    ));
                }
                treatment_observation_count += 1;
            }
            PlannedRunKind::CenterPointControl => {
                let block_sum = center_sum_by_block
                    .entry(run.block_id.clone())
                    .or_insert(0.0);
                *block_sum += outcome;
                if !block_sum.is_finite() {
                    return Err(ScreeningAnalysisError::new(
                        "observations.outcome_value",
                        "center-point total overflow",
                    ));
                }
                *center_count_by_block.entry(run.block_id.clone()).or_insert(0) += 1;
                center_point_observation_count += 1;
            }
        }
    }

    if treatment_observation_count == 0 {
        return Err(ScreeningAnalysisError::new(
            "observations",
            "at least one factorial treatment observation is required",
        ));
    }

    let factorial_replicates =
        request.blocks.len() * request.replicates_per_setting_per_block as usize;
    let denominator = (factorial_replicates * (1_usize << (factor_count - 1))) as f64;
    let mut factorial_effects = Vec::with_capacity(combination_count - 1);

    for effect_mask in 1..combination_count {
        let mut contrast_sum = 0.0_f64;
        for row in 0..combination_count {
            let standard_order = (row + 1) as u32;
            let mut row_sum = 0.0_f64;
            let mut row_count = 0_u32;
            for block in &request.blocks {
                let key = (block.block_id.clone(), standard_order);
                let observed_sum = *cell_sum.get(&key).ok_or_else(|| {
                    ScreeningAnalysisError::new(
                        "observations.factorial_coverage",
                        "a treatment cell is missing from the analysis data",
                    )
                })?;
                let observed_count = *cell_count.get(&key).ok_or_else(|| {
                    ScreeningAnalysisError::new(
                        "observations.factorial_coverage",
                        "a treatment cell count is missing",
                    )
                })?;
                if observed_count != request.replicates_per_setting_per_block as u32 {
                    return Err(ScreeningAnalysisError::new(
                        "observations.factorial_coverage",
                        "treatment cell count does not match the preregistered replicate count",
                    ));
                }
                row_sum += observed_sum;
                row_count += observed_count;
            }
            if row_count == 0 {
                return Err(ScreeningAnalysisError::new(
                    "observations.factorial_coverage",
                    "no observations for one factorial row",
                ));
            }
            let row_sum_nonfinite = !row_sum.is_finite();
            if row_sum_nonfinite {
                return Err(ScreeningAnalysisError::new(
                    "observations.outcome_value",
                    "factorial row total overflow",
                ));
            }
            let row_mean = row_sum / row_count as f64;
            let high_count = (row as u32 & effect_mask as u32).count_ones();
            let effect_order = effect_mask.count_ones();
            let sign = if high_count % 2 == effect_order % 2 { 1.0 } else { -1.0 };
            contrast_sum += sign * row_mean * factorial_replicates as f64;
            if !contrast_sum.is_finite() {
                return Err(ScreeningAnalysisError::new(
                    "observations.outcome_value",
                    "factorial contrast overflow",
                ));
            }
        }

        let effect_estimate = contrast_sum / denominator;
        if !effect_estimate.is_finite() {
            return Err(ScreeningAnalysisError::new(
                "factorial_effects",
                "effect estimate is non-finite",
            ));
        }
        let effect_factor_indices: Vec<usize> = (0..factor_count)
            .filter(|factor_index| effect_mask & (1_usize << *factor_index) != 0)
            .collect();
        let factor_ids = effect_factor_indices
            .iter()
            .map(|index| factors[*index].factor_id.clone())
            .collect::<Vec<_>>();
        let effect_order = effect_mask.count_ones();
        let effect_id = effect_factor_indices
            .iter()
            .map(|index| format!("{:02}", index + 1))
            .collect::<Vec<_>>()
            .join("-");
        let kind = match effect_order {
            1 => FactorialEffectOrder::MainEffect,
            2 => FactorialEffectOrder::TwoFactorInteraction,
            _ => FactorialEffectOrder::HigherOrderInteraction,
        };
        factorial_effects.push(FactorialEffectEstimate {
            effect_id: format!("effect-{effect_id}"),
            factor_ids,
            order: effect_order,
            kind,
            effect_estimate,
            unit: endpoint.unit.clone(),
            independent_complete_factorial_replicates: factorial_replicates as u32,
            observation_count: treatment_observation_count,
        });
    }

    let factorial_treatment_mean = factorial_total / treatment_observation_count as f64;
    if !factorial_treatment_mean.is_finite() {
        return Err(ScreeningAnalysisError::new(
            "factorial_treatment_mean",
            "mean is non-finite",
        ));
    }

    let mut block_center_point_diagnostics = Vec::new();
    if request.center_point_runs_per_block > 0 {
        for block in &request.blocks {
            let center_count = *center_count_by_block.get(&block.block_id).unwrap_or(&0);
            let treatment_count = *treatment_count_by_block.get(&block.block_id).unwrap_or(&0);
            if center_count != request.center_point_runs_per_block as u32
                || treatment_count
                    != (combination_count * request.replicates_per_setting_per_block as usize) as u32
            {
                return Err(ScreeningAnalysisError::new(
                    "observations.block_coverage",
                    "each block must contain every preregistered treatment replicate and center-point outcome",
                ));
            }
            let center_sum = *center_sum_by_block.get(&block.block_id).ok_or_else(|| {
                ScreeningAnalysisError::new(
                    "observations.center_points",
                    "center-point data missing for a block",
                )
            })?;
            let treatment_sum = treatment_sum_by_block[&block.block_id];
            let center_mean = center_sum / center_count as f64;
            let treatment_mean = treatment_sum / treatment_count as f64;
            let difference = center_mean - treatment_mean;
            if !center_mean.is_finite() || !treatment_mean.is_finite() || !difference.is_finite() {
                return Err(ScreeningAnalysisError::new(
                    "observations.center_points",
                    "center-point diagnostic is non-finite",
                ));
            }
            block_center_point_diagnostics.push(BlockCenterPointDiagnostic {
                block_id: block.block_id.clone(),
                factorial_treatment_mean: treatment_mean,
                center_point_mean: center_mean,
                center_point_minus_factorial_mean: difference,
                unit: endpoint.unit.clone(),
                center_point_observation_count: center_count,
                factorial_treatment_observation_count: treatment_count,
            });
        }
    }

    Ok(ScreeningResponseAnalysis {
        algorithm_id: RESPONSE_ANALYSIS_ALGORITHM_ID.into(),
        experiment_id: request.experiment_id.clone(),
        preregistration_id: request.preregistration_id.clone(),
        input_snapshot_id: request.input_snapshot_id.clone(),
        analysis_plan_id: request.analysis_plan_id.clone(),
        endpoint_id: endpoint.endpoint_id.clone(),
        endpoint_unit: endpoint.unit.clone(),
        design_verification,
        request_snapshot: request.clone(),
        schedule_snapshot: plan.runs.clone(),
        observations_in_run_order: ordered_observations,
        treatment_observation_count,
        center_point_observation_count,
        factorial_effects,
        block_center_point_diagnostics,
        factorial_treatment_mean,
        all_observation_count: observations.len() as u32,
        scope_note: "descriptive factorial contrasts and block-wise center-point differences only; no p-values, confidence intervals, causal claims, safety determination, field-trial authorization, product release, or agronomic-efficacy claim".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening_design::{
        BenchScaleReview, BenchScaleReviewStatus, ExperimentBlock, FactorRandomizationClass,
        PrimaryEndpoint, ScreeningFactor, ScreeningDesignRequest,
    };

    fn evidence(id: &str, kind: EvidenceKind) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind,
        }
    }

    fn request() -> ScreeningDesignRequest {
        let mut request = ScreeningDesignRequest {
            experiment_id: "soil-analysis-001".into(),
            preregistration_id: "prereg-analysis-v1".into(),
            input_snapshot_id: "analysis-inputs-v1".into(),
            protocol_id: "analysis-protocol-v1".into(),
            objective: "Describe factorial effects on dry-basis char yield".into(),
            primary_hypothesis: "At least one factor has a non-zero main effect".into(),
            analysis_plan_id: "analysis-plan-v1".into(),
            randomization_seed: 0x5eed,
            factors: vec![
                ScreeningFactor {
                    factor_id: "temperature".into(),
                    label: "Temperature".into(),
                    unit: "degree_C".into(),
                    low_value: 400.0,
                    high_value: 500.0,
                    max_absolute_deviation: 2.0,
                    deviation_tolerance_evidence: evidence("temp-tolerance", EvidenceKind::Literature),
                    randomization_class: FactorRandomizationClass::RandomizablePerRun,
                    low_level_evidence: evidence("temp-low", EvidenceKind::Literature),
                    high_level_evidence: evidence("temp-high", EvidenceKind::Literature),
                },
                ScreeningFactor {
                    factor_id: "residence".into(),
                    label: "Residence time".into(),
                    unit: "minute".into(),
                    low_value: 10.0,
                    high_value: 30.0,
                    max_absolute_deviation: 1.0,
                    deviation_tolerance_evidence: evidence("res-tolerance", EvidenceKind::Literature),
                    randomization_class: FactorRandomizationClass::RandomizablePerRun,
                    low_level_evidence: evidence("res-low", EvidenceKind::Literature),
                    high_level_evidence: evidence("res-high", EvidenceKind::Literature),
                },
            ],
            blocks: vec![
                ExperimentBlock {
                    block_id: "day-1".into(),
                    description: "First operating day".into(),
                    evidence: evidence("day1", EvidenceKind::Measured),
                },
                ExperimentBlock {
                    block_id: "day-2".into(),
                    description: "Second operating day".into(),
                    evidence: evidence("day2", EvidenceKind::Measured),
                },
            ],
            replicates_per_setting_per_block: 1,
            center_point_runs_per_block: 3,
            primary_endpoint: PrimaryEndpoint {
                endpoint_id: "dry-char-yield".into(),
                description: "Dry char mass divided by dry feedstock mass".into(),
                unit: "kg_per_kg_dry_feedstock".into(),
                measurement_method_id: "gravimetric-method-v1".into(),
                minimum_practically_meaningful_difference: 0.03,
                difference_rationale_evidence: evidence("mpmd-rationale", EvidenceKind::Literature),
            },
            bench_scale_review: BenchScaleReview {
                status: BenchScaleReviewStatus::ApprovedForThisBenchScaleProtocol,
                protocol_id: "analysis-protocol-v1".into(),
                reviewed_input_snapshot_id: "analysis-inputs-v1".into(),
                reviewed_design_sha256: String::new(),
                review_id: "review-analysis-v1".into(),
                reviewer_role: "qualified bench-scale reviewer".into(),
                evidence: evidence("review-analysis", EvidenceKind::Measured),
            },
        };
        request.bench_scale_review.reviewed_design_sha256 =
            crate::screening_design::screening_design_sha256(&request).unwrap();
        request
    }

    fn observations(plan: &ScreeningDesignPlan) -> Vec<ScreeningResponseObservation> {
        plan.runs
            .iter()
            .map(|run| {
                let base = if run.block_id == "day-1" { 0.0 } else { 100.0 };
                let outcome_value = match run.kind {
                    PlannedRunKind::CenterPointControl => 15.0 + base,
                    PlannedRunKind::FactorialTreatment => {
                        let x_a = run.settings[0].coded_level as f64;
                        let x_b = run.settings[1].coded_level as f64;
                        10.0 + 2.0 * x_a + 3.0 * x_b + 4.0 * x_a * x_b + base
                    }
                };
                ScreeningResponseObservation {
                    observation_id: format!("obs-{}", run.run_id),
                    run_id: run.run_id.clone(),
                    actual_factor_settings: run.settings.iter().map(|setting| {
                        ObservedFactorSetting {
                            factor_id: setting.factor_id.clone(),
                            value: setting.value,
                            unit: setting.unit.clone(),
                            evidence: evidence(
                                &format!("achieved-setting-{}-{}", run.run_id, setting.factor_id),
                                EvidenceKind::Measured,
                            ),
                        }
                    }).collect(),
                    endpoint_id: "dry-char-yield".into(),
                    outcome_value,
                    outcome_unit: "kg_per_kg_dry_feedstock".into(),
                    measurement_method_id: "gravimetric-method-v1".into(),
                    evidence: evidence(
                        &format!("lab-result-{}", run.run_id),
                        EvidenceKind::Measured,
                    ),
                }
            })
            .collect()
    }

    #[test]
    fn measured_data_produce_expected_factorial_contrasts_and_center_diagnostics() {
        let plan = crate::screening_design::generate_screening_design(&request()).unwrap();
        let data = observations(&plan);
        let result = analyze_screening_responses(&plan, &data).unwrap();

        assert_eq!(result.treatment_observation_count, 8);
        assert_eq!(result.center_point_observation_count, 6);
        assert_eq!(result.all_observation_count, 14);
        assert_eq!(result.factorial_effects.len(), 3);
        let a = result.factorial_effects.iter()
            .find(|effect| effect.factor_ids == vec!["temperature".to_string()])
            .unwrap();
        let b = result.factorial_effects.iter()
            .find(|effect| effect.factor_ids == vec!["residence".to_string()])
            .unwrap();
        let ab = result.factorial_effects.iter()
            .find(|effect| {
                effect.factor_ids
                    == vec!["temperature".to_string(), "residence".to_string()]
            })
            .unwrap();
        assert!((a.effect_estimate - 4.0).abs() < 1e-12);
        assert!((b.effect_estimate - 6.0).abs() < 1e-12);
        assert!((ab.effect_estimate - 8.0).abs() < 1e-12);
        assert!(result.block_center_point_diagnostics.iter()
            .all(|diagnostic| (diagnostic.center_point_minus_factorial_mean - 5.0).abs() < 1e-12));
        assert_eq!(result.observations_in_run_order[0].run_id, plan.runs[0].run_id);
        assert_eq!(result.schedule_snapshot, plan.runs);
        assert!(result.scope_note.contains("no p-values"));
    }

    #[test]
    fn missing_duplicate_unknown_and_scenario_observations_are_rejected() {
        let plan = crate::screening_design::generate_screening_design(&request()).unwrap();
        let mut data = observations(&plan);
        data.pop();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[1].observation_id = data[0].observation_id.clone();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].run_id = "unknown-run".into();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].evidence.kind = EvidenceKind::Scenario;
        assert!(analyze_screening_responses(&plan, &data).is_err());
    }

    #[test]
    fn measured_achieved_settings_must_match_preregistered_tolerances() {
        let plan = crate::screening_design::generate_screening_design(&request()).unwrap();
        let mut data = observations(&plan);
        data[0].actual_factor_settings[0].value += 1.0;
        assert!(analyze_screening_responses(&plan, &data).is_ok());

        data = observations(&plan);
        data[0].actual_factor_settings[0].value += 3.0;
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].actual_factor_settings[0].evidence.kind = EvidenceKind::Scenario;
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].actual_factor_settings[0].unit = "percent".into();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].actual_factor_settings[0].factor_id = "unknown-factor".into();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].actual_factor_settings[1].factor_id =
            data[0].actual_factor_settings[0].factor_id.clone();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].actual_factor_settings[0].value = f64::NAN;
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].actual_factor_settings.pop();
        assert!(analyze_screening_responses(&plan, &data).is_err());
    }

    #[test]
    fn outcome_unit_method_endpoint_and_finiteness_must_match_preregistration() {
        let plan = crate::screening_design::generate_screening_design(&request()).unwrap();
        let mut data = observations(&plan);
        data[0].outcome_unit = "percent".into();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].measurement_method_id = "different-method".into();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].endpoint_id = "secondary-endpoint".into();
        assert!(analyze_screening_responses(&plan, &data).is_err());

        data = observations(&plan);
        data[0].outcome_value = f64::NAN;
        assert!(analyze_screening_responses(&plan, &data).is_err());
    }

    #[test]
    fn corrupted_schedule_is_rejected_before_response_analysis() {
        let mut plan = crate::screening_design::generate_screening_design(&request()).unwrap();
        let treatment = plan.runs.iter_mut()
            .find(|run| run.kind == PlannedRunKind::FactorialTreatment)
            .unwrap();
        treatment.settings[0].value += 1.0;
        let data = observations(&plan);
        assert!(analyze_screening_responses(&plan, &data).is_err());
    }

    #[test]
    fn nonfinite_aggregates_and_bad_cell_replicates_fail_closed() {
        let mut plan = crate::screening_design::generate_screening_design(&request()).unwrap();
        // A structurally valid schedule is required, so overflow is tested via extreme,
        // finite responses whose summed treatment total cannot be represented.
        let mut data = observations(&plan);
        for observation in &mut data {
            if observation.run_id.contains("day-1") {
                observation.outcome_value = f64::MAX;
            }
        }
        assert!(analyze_screening_responses(&plan, &data).is_err());

        plan.request_snapshot.replicates_per_setting_per_block = 2;
        let data = observations(&plan);
        assert!(analyze_screening_responses(&plan, &data).is_err());
    }
}
