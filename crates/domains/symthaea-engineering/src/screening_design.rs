// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible, preregistration-first two-level full-factorial screening plans.
//!
//! This module generates run schedules, not predictions. It supports two to six numeric
//! factors, complete blocks, replicated treatment settings, deterministic randomization,
//! and optional center-point controls. It does not calculate statistical power, optimize
//! a design under budget, estimate causal effects, approve a field trial, or certify safety.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use symthaea_agribot::soil_process::{EvidenceKind, EvidenceRef};

const DESIGN_ALGORITHM_ID: &str = "full-factorial-2level-blocked-splitmix64-fy-v1";
const MIN_FACTORS: usize = 2;
const MAX_FACTORS: usize = 6;
const MIN_BLOCKS: usize = 2;
const MAX_BLOCKS: usize = 8;
const MAX_REPLICATES_PER_SETTING_PER_BLOCK: u8 = 2;
const MAX_TOTAL_RUNS: usize = 1_100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreeningDesignError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl ScreeningDesignError {
    fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl std::fmt::Display for ScreeningDesignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl std::error::Error for ScreeningDesignError {}

fn nonempty(value: &str, field: &'static str) -> Result<(), ScreeningDesignError> {
    if value.trim().is_empty() {
        return Err(ScreeningDesignError::new(field, "cannot be empty"));
    }
    Ok(())
}

fn evidence(evidence: &EvidenceRef, field: &'static str) -> Result<(), ScreeningDesignError> {
    nonempty(&evidence.evidence_id, field)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactorRandomizationClass {
    /// Each setting can be assigned to individual runs in the randomized complete block.
    RandomizablePerRun,
    /// A hard-to-change factor or equipment constraint requires a restricted/split-plot design.
    RestrictedOrHardToChange,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningFactor {
    /// Stable identifier used in the preregistration and analysis dataset.
    pub factor_id: String,
    pub label: String,
    pub unit: String,
    /// Lower tested factor setting. This is a numeric, quantitative factor.
    pub low_value: f64,
    /// Higher tested factor setting. This is a numeric, quantitative factor.
    pub high_value: f64,
    /// Randomization class must be established for the actual apparatus and protocol.
    pub randomization_class: FactorRandomizationClass,
    /// Evidence for the lower level and its applicability to this process.
    pub low_level_evidence: EvidenceRef,
    /// Evidence for the higher level and its applicability to this process.
    pub high_level_evidence: EvidenceRef,
}

impl ScreeningFactor {
    fn validate(&self) -> Result<(), ScreeningDesignError> {
        nonempty(&self.factor_id, "factors.factor_id")?;
        nonempty(&self.label, "factors.label")?;
        nonempty(&self.unit, "factors.unit")?;
        if !self.low_value.is_finite() || !self.high_value.is_finite() {
            return Err(ScreeningDesignError::new(
                "factors.levels",
                "factor levels must be finite",
            ));
        }
        if self.low_value >= self.high_value {
            return Err(ScreeningDesignError::new(
                "factors.levels",
                "low_value must be strictly less than high_value",
            ));
        }
        if self.randomization_class != FactorRandomizationClass::RandomizablePerRun {
            return Err(ScreeningDesignError::new(
                "factors.randomization_class",
                "this generator requires per-run randomization; restricted or hard-to-change factors need a separately specified split-plot/restricted-randomization design",
            ));
        }
        evidence(&self.low_level_evidence, "factors.low_level_evidence")?;
        evidence(&self.high_level_evidence, "factors.high_level_evidence")?;
        Ok(())
    }

    fn midpoint(&self) -> f64 {
        // Half-sum avoids overflow in high_value - low_value for large finite bounds.
        self.low_value / 2.0 + self.high_value / 2.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimaryEndpoint {
    pub endpoint_id: String,
    pub description: String,
    pub unit: String,
    pub measurement_method_id: String,
    /// Minimum practically meaningful difference, declared before treatment results.
    /// This is not a minimum detectable effect or a claim that a design is adequately powered.
    pub minimum_practically_meaningful_difference: f64,
    pub difference_rationale_evidence: EvidenceRef,
}

impl PrimaryEndpoint {
    fn validate(&self) -> Result<(), ScreeningDesignError> {
        nonempty(&self.endpoint_id, "primary_endpoint.endpoint_id")?;
        nonempty(&self.description, "primary_endpoint.description")?;
        nonempty(&self.unit, "primary_endpoint.unit")?;
        nonempty(
            &self.measurement_method_id,
            "primary_endpoint.measurement_method_id",
        )?;
        if !self.minimum_practically_meaningful_difference.is_finite()
            || self.minimum_practically_meaningful_difference <= 0.0
        {
            return Err(ScreeningDesignError::new(
                "primary_endpoint.minimum_practically_meaningful_difference",
                "must be finite and greater than zero",
            ));
        }
        evidence(
            &self.difference_rationale_evidence,
            "primary_endpoint.difference_rationale_evidence",
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentBlock {
    /// Stable nuisance-stratum identifier, such as day, feedstock lot, or soil batch.
    pub block_id: String,
    pub description: String,
    pub evidence: EvidenceRef,
}

impl ExperimentBlock {
    fn validate(&self) -> Result<(), ScreeningDesignError> {
        nonempty(&self.block_id, "blocks.block_id")?;
        nonempty(&self.description, "blocks.description")?;
        evidence(&self.evidence, "blocks.evidence")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchScaleReviewStatus {
    ApprovedForThisBenchScaleProtocol,
    Rejected,
    Unknown,
}

/// Declared reviewer decision for a named bench-scale protocol. The planner validates
/// status, evidence-ID presence, non-scenario evidence class, and exact protocol identity;
/// it does not resolve, authenticate, or independently verify the referenced review record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchScaleReview {
    pub status: BenchScaleReviewStatus,
    pub protocol_id: String,
    /// Must match the exact request input snapshot containing levels and endpoint.
    pub reviewed_input_snapshot_id: String,
    pub review_id: String,
    pub reviewer_role: String,
    pub evidence: EvidenceRef,
}

/// Complete, preregistration-first request. This version accepts quantitative factors only.
/// Treatment allocation, product-quality release, field application, and agronomic claims
/// remain separate decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningDesignRequest {
    pub experiment_id: String,
    pub preregistration_id: String,
    pub input_snapshot_id: String,
    pub protocol_id: String,
    pub objective: String,
    pub primary_hypothesis: String,
    pub analysis_plan_id: String,
    /// A user-supplied seed. Same request and algorithm version produce the same schedule.
    pub randomization_seed: u64,
    pub factors: Vec<ScreeningFactor>,
    /// Complete blocks: every factorial treatment setting appears within every block.
    pub blocks: Vec<ExperimentBlock>,
    /// Independent experimental units per treatment setting in each block.
    /// Repeated measurements of the same unit do not count as replicates.
    pub replicates_per_setting_per_block: u8,
    /// Either 0 (no center points) or 3..=5 control runs per block. These controls
    /// use the factor midpoints, begin/end the block schedule, and are evenly dispersed.
    pub center_point_runs_per_block: u8,
    pub primary_endpoint: PrimaryEndpoint,
    pub bench_scale_review: BenchScaleReview,
}

impl ScreeningDesignRequest {
    fn validate(&self) -> Result<usize, ScreeningDesignError> {
        nonempty(&self.experiment_id, "experiment_id")?;
        nonempty(&self.preregistration_id, "preregistration_id")?;
        nonempty(&self.input_snapshot_id, "input_snapshot_id")?;
        nonempty(&self.protocol_id, "protocol_id")?;
        nonempty(&self.objective, "objective")?;
        nonempty(&self.primary_hypothesis, "primary_hypothesis")?;
        nonempty(&self.analysis_plan_id, "analysis_plan_id")?;

        if !(MIN_FACTORS..=MAX_FACTORS).contains(&self.factors.len()) {
            return Err(ScreeningDesignError::new(
                "factors",
                "two-level full factorial requires 2 to 6 factors in this implementation",
            ));
        }
        let mut factor_ids = HashSet::new();
        for factor in &self.factors {
            factor.validate()?;
            if !factor_ids.insert(factor.factor_id.as_str()) {
                return Err(ScreeningDesignError::new(
                    "factors.factor_id",
                    "factor IDs must be unique",
                ));
            }
        }

        if !(MIN_BLOCKS..=MAX_BLOCKS).contains(&self.blocks.len()) {
            return Err(ScreeningDesignError::new(
                "blocks",
                "use 2 to 8 complete blocks; document the nuisance stratum each block represents",
            ));
        }
        let mut block_ids = HashSet::new();
        for block in &self.blocks {
            block.validate()?;
            if !block_ids.insert(block.block_id.as_str()) {
                return Err(ScreeningDesignError::new(
                    "blocks.block_id",
                    "block IDs must be unique",
                ));
            }
        }
        if self.replicates_per_setting_per_block == 0
            || self.replicates_per_setting_per_block > MAX_REPLICATES_PER_SETTING_PER_BLOCK
        {
            return Err(ScreeningDesignError::new(
                "replicates_per_setting_per_block",
                "must be 1 or 2 independently executed experimental units per setting in each block",
            ));
        }
        if self.center_point_runs_per_block != 0
            && !(3..=5).contains(&self.center_point_runs_per_block)
        {
            return Err(ScreeningDesignError::new(
                "center_point_runs_per_block",
                "must be 0 or between 3 and 5",
            ));
        }

        self.primary_endpoint.validate()?;
        nonempty(&self.bench_scale_review.protocol_id, "bench_scale_review.protocol_id")?;
        nonempty(
            &self.bench_scale_review.reviewed_input_snapshot_id,
            "bench_scale_review.reviewed_input_snapshot_id",
        )?;
        if self.bench_scale_review.reviewed_input_snapshot_id != self.input_snapshot_id {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.reviewed_input_snapshot_id",
                "review must reference the exact input snapshot containing the factor ranges and endpoint",
            ));
        }
        nonempty(&self.bench_scale_review.review_id, "bench_scale_review.review_id")?;
        nonempty(&self.bench_scale_review.reviewer_role, "bench_scale_review.reviewer_role")?;
        evidence(&self.bench_scale_review.evidence, "bench_scale_review.evidence")?;
        if self.bench_scale_review.protocol_id != self.protocol_id {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.protocol_id",
                "review must apply to the exact protocol ID in this request",
            ));
        }
        if self.bench_scale_review.status
            != BenchScaleReviewStatus::ApprovedForThisBenchScaleProtocol
        {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.status",
                "run plan withheld unless a reviewer approves this exact bench-scale protocol",
            ));
        }
        if self.bench_scale_review.evidence.kind == EvidenceKind::Scenario {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.evidence",
                "scenario evidence cannot establish a bench-scale safety review",
            ));
        }

        let combinations = 1_usize << self.factors.len();
        let treatment_runs = combinations
            .checked_mul(self.blocks.len())
            .and_then(|count| count.checked_mul(self.replicates_per_setting_per_block as usize))
            .ok_or_else(|| ScreeningDesignError::new("run_count", "treatment run count overflow"))?;
        let center_runs = (self.center_point_runs_per_block as usize)
            .checked_mul(self.blocks.len())
            .ok_or_else(|| {
                ScreeningDesignError::new("run_count", "center point run count overflow")
            })?;
        let total_runs = treatment_runs
            .checked_add(center_runs)
            .ok_or_else(|| ScreeningDesignError::new("run_count", "total run count overflow"))?;
        if total_runs > MAX_TOTAL_RUNS {
            return Err(ScreeningDesignError::new(
                "run_count",
                "design exceeds the 1,100-run implementation bound",
            ));
        }
        Ok(total_runs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlannedRunKind {
    FactorialTreatment,
    CenterPointControl,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorSetting {
    pub factor_id: String,
    /// -1 is low, +1 is high, and 0 is midpoint center-point setting.
    pub coded_level: i8,
    pub value: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedRun {
    pub run_id: String,
    /// One-based sequence across the full scheduled experiment.
    pub run_order: u32,
    pub block_id: String,
    pub kind: PlannedRunKind,
    /// Standard-order factorial row (one-based), absent for center-point controls.
    pub standard_order: Option<u32>,
    /// Replicate within the specified block, absent for center-point controls.
    pub replicate_index: Option<u8>,
    pub settings: Vec<FactorSetting>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningDesignPlan {
    pub algorithm_id: String,
    /// Full copy of the preregistered inputs used to generate this schedule.
    pub request_snapshot: ScreeningDesignRequest,
    pub treatment_combination_count: u32,
    pub treatment_run_count: u32,
    pub center_point_run_count: u32,
    pub total_planned_run_count: u32,
    pub runs: Vec<PlannedRun>,
    pub interpretation_note: String,
}

fn next_random(state: &mut u64) -> u64 {
    // SplitMix64: stable, non-cryptographic, reproducible pseudo-random stream.
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn bounded_random(state: &mut u64, bound: usize) -> usize {
    let bound = bound as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = next_random(state);
        if value >= threshold {
            return (value % bound) as usize;
        }
    }
}

fn shuffle<T>(values: &mut [T], state: &mut u64) {
    for index in (1..values.len()).rev() {
        let other = bounded_random(state, index + 1);
        values.swap(index, other);
    }
}

fn settings_for(
    factors: &[ScreeningFactor],
    standard_row: usize,
    center_point: bool,
) -> Vec<FactorSetting> {
    factors
        .iter()
        .enumerate()
        .map(|(factor_index, factor)| {
            let (coded_level, value) = if center_point {
                (0, factor.midpoint())
            } else if standard_row & (1_usize << factor_index) == 0 {
                (-1, factor.low_value)
            } else {
                (1, factor.high_value)
            };
            FactorSetting {
                factor_id: factor.factor_id.clone(),
                coded_level,
                value,
                unit: factor.unit.clone(),
            }
        })
        .collect()
}

/// Generate a proposed preregistered blocked, randomized two-level full-factorial schedule.
/// Treatment runs and center-point controls, when enabled, are jointly shuffled within
/// each complete block. Center points use numeric midpoints. Including controls in the
/// randomization avoids systematically confounding them with early/late run-order drift.
/// Review status and record references are checked structurally but are not authenticated
/// against an independent evidence service. This function does not authorize execution.
pub fn generate_screening_design(
    request: &ScreeningDesignRequest,
) -> Result<ScreeningDesignPlan, ScreeningDesignError> {
    let total_runs = request.validate()?;
    let factor_count = request.factors.len();
    let combination_count = 1_usize << factor_count;
    let treatment_runs_per_block =
        combination_count * request.replicates_per_setting_per_block as usize;
    let center_count = request.center_point_runs_per_block as usize;
    let mut rng_state = request.randomization_seed;
    let mut runs = Vec::with_capacity(total_runs);
    let mut next_order = 1_u32;
    let mut treatment_run_count = 0_u32;
    let mut center_run_count = 0_u32;

    for block in &request.blocks {
        // Randomize treatment runs and center-point controls together inside each block.
        // Fixing controls to start/end positions would alias them with run-order drift.
        let mut schedule = Vec::with_capacity(treatment_runs_per_block + center_count);
        for standard_row in 0..combination_count {
            for replicate in 1..=request.replicates_per_setting_per_block {
                schedule.push(Some((standard_row, replicate)));
            }
        }
        for _ in 0..center_count {
            schedule.push(None);
        }
        shuffle(&mut schedule, &mut rng_state);

        for scheduled_item in schedule {
            let (kind, standard_order, replicate_index, settings) =
                if let Some((standard_row, replicate)) = scheduled_item {
                    treatment_run_count += 1;
                    (
                        PlannedRunKind::FactorialTreatment,
                        Some((standard_row + 1) as u32),
                        Some(replicate),
                        settings_for(&request.factors, standard_row, false),
                    )
                } else {
                    center_run_count += 1;
                    (
                        PlannedRunKind::CenterPointControl,
                        None,
                        None,
                        settings_for(&request.factors, 0, true),
                    )
                };
            let run_id = format!(
                "{}-{}-{:04}",
                request.experiment_id, block.block_id, next_order
            );
            runs.push(PlannedRun {
                run_id,
                run_order: next_order,
                block_id: block.block_id.clone(),
                kind,
                standard_order,
                replicate_index,
                settings,
            });
            next_order += 1;
        }
    }

    debug_assert_eq!(runs.len(), total_runs);
    debug_assert_eq!(
        treatment_run_count as usize,
        combination_count
            * request.blocks.len()
            * request.replicates_per_setting_per_block as usize
    );
    debug_assert_eq!(center_run_count as usize, center_count * request.blocks.len());

    Ok(ScreeningDesignPlan {
        algorithm_id: DESIGN_ALGORITHM_ID.to_string(),
        request_snapshot: request.clone(),
        treatment_combination_count: combination_count as u32,
        treatment_run_count,
        center_point_run_count: center_run_count,
        total_planned_run_count: runs.len() as u32,
        runs,
        interpretation_note: "deterministic bench-scale screening schedule only; not a power analysis, response-surface model, product-quality release, field-trial authorization, application-rate recommendation, or proof of agronomic efficacy. Block effects and center-point behavior must be included in the preregistered analysis."
            .into(),
    })
}



#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningDesignVerificationReceipt {
    pub experiment_id: String,
    pub input_snapshot_id: String,
    pub algorithm_id: String,
    pub verified_run_count: u32,
    pub verified_block_count: u32,
    pub verified_factorial_combination_count: u32,
    pub verified_treatment_replicate_cell_count: u32,
    pub verified_center_point_count: u32,
    /// Structural design checks passed. This is not evidence authentication or scientific validation.
    pub scope_note: String,
}

/// Recheck a generated schedule's structural invariants without calling the generator.
/// This validates the serialized design artifact, but it does not authenticate its cited
/// evidence, independently reproduce the software in another trust domain, or authorize execution.
pub fn verify_screening_design(
    plan: &ScreeningDesignPlan,
) -> Result<ScreeningDesignVerificationReceipt, ScreeningDesignError> {
    let request = &plan.request_snapshot;
    let expected_total = request.validate()?;
    if plan.algorithm_id != DESIGN_ALGORITHM_ID {
        return Err(ScreeningDesignError::new(
            "algorithm_id",
            "unknown or mismatched design-generation algorithm",
        ));
    }
    let factor_count = request.factors.len();
    let combination_count = 1_usize << factor_count;
    let expected_treatment_runs = combination_count
        * request.blocks.len()
        * request.replicates_per_setting_per_block as usize;
    let center_count = request.center_point_runs_per_block as usize;
    let expected_center_runs = center_count * request.blocks.len();

    if plan.treatment_combination_count as usize != combination_count
        || plan.treatment_run_count as usize != expected_treatment_runs
        || plan.center_point_run_count as usize != expected_center_runs
        || plan.total_planned_run_count as usize != expected_total
        || plan.runs.len() != expected_total
    {
        return Err(ScreeningDesignError::new(
            "plan.run_counts",
            "stored run counts do not match the declared design",
        ));
    }

    let treatment_runs_per_block =
        combination_count * request.replicates_per_setting_per_block as usize;
    let mut run_ids = HashSet::new();
    let mut seen_cells: HashSet<(String, u32, u8)> = HashSet::new();
    let mut current_block_index = 0_usize;
    let mut local_position = 0_usize;
    let mut current_block_run_count = 0_usize;
    let mut previous_block_id: Option<&str> = None;
    let mut verified_treatments = 0_u32;
    let mut verified_centers = 0_u32;

    for (index, run) in plan.runs.iter().enumerate() {
        let expected_order = (index + 1) as u32;
        if run.run_order != expected_order {
            return Err(ScreeningDesignError::new(
                "runs.run_order",
                "run order must be contiguous and one-based",
            ));
        }
        let expected_id = format!(
            "{}-{}-{:04}",
            request.experiment_id, run.block_id, run.run_order
        );
        if run.run_id != expected_id || !run_ids.insert(run.run_id.as_str()) {
            return Err(ScreeningDesignError::new(
                "runs.run_id",
                "run IDs must match the declared naming scheme and be unique",
            ));
        }

        let block_index = request
            .blocks
            .iter()
            .position(|block| block.block_id == run.block_id)
            .ok_or_else(|| {
                ScreeningDesignError::new("runs.block_id", "run references an unknown block")
            })?;
        if previous_block_id != Some(run.block_id.as_str()) {
            if previous_block_id.is_some() {
                if local_position != treatment_runs_per_block + center_count {
                    return Err(ScreeningDesignError::new(
                        "runs.block_sequence",
                        "previous block is incomplete",
                    ));
                }
                if block_index != current_block_index + 1 {
                    return Err(ScreeningDesignError::new(
                        "runs.block_sequence",
                        "blocks must appear in preregistered order and remain contiguous",
                    ));
                }
            } else if block_index != 0 {
                return Err(ScreeningDesignError::new(
                    "runs.block_sequence",
                    "first scheduled block must be the first preregistered block",
                ));
            }
            current_block_index = block_index;
            local_position = 0;
            current_block_run_count = 0;
            previous_block_id = Some(run.block_id.as_str());
        }

        if run.settings.len() != factor_count {
            return Err(ScreeningDesignError::new(
                "runs.settings",
                "each run must carry one setting for every declared factor",
            ));
        }

        match run.kind {
            PlannedRunKind::FactorialTreatment => {
                let standard_order = run.standard_order.ok_or_else(|| {
                    ScreeningDesignError::new(
                        "runs.standard_order",
                        "treatment requires a standard-order row",
                    )
                })?;
                let replicate = run.replicate_index.ok_or_else(|| {
                    ScreeningDesignError::new(
                        "runs.replicate_index",
                        "treatment requires a replicate index",
                    )
                })?;
                if standard_order == 0 || standard_order as usize > combination_count {
                    return Err(ScreeningDesignError::new(
                        "runs.standard_order",
                        "treatment standard-order row is outside the factorial design",
                    ));
                }
                if replicate == 0 || replicate > request.replicates_per_setting_per_block {
                    return Err(ScreeningDesignError::new(
                        "runs.replicate_index",
                        "replicate index is outside the preregistered range",
                    ));
                }
                let cell = (run.block_id.clone(), standard_order, replicate);
                if !seen_cells.insert(cell) {
                    return Err(ScreeningDesignError::new(
                        "runs.factorial_coverage",
                        "a block contains a duplicate treatment/replicate cell",
                    ));
                }
                let row = standard_order as usize - 1;
                for (factor_index, (setting, factor)) in
                    run.settings.iter().zip(&request.factors).enumerate()
                {
                    let is_high = row & (1_usize << factor_index) != 0;
                    let expected_code = if is_high { 1 } else { -1 };
                    let expected_value = if is_high { factor.high_value } else { factor.low_value };
                    if setting.factor_id != factor.factor_id
                        || setting.unit != factor.unit
                        || setting.coded_level != expected_code
                        || setting.value != expected_value
                    {
                        return Err(ScreeningDesignError::new(
                            "runs.settings",
                            "treatment factor IDs, units, coded levels, or values do not match the design row",
                        ));
                    }
                }
                verified_treatments += 1;
            }
            PlannedRunKind::CenterPointControl => {
                if run.standard_order.is_some() || run.replicate_index.is_some() {
                    return Err(ScreeningDesignError::new(
                        "runs.center_point",
                        "center-point control must not masquerade as a factorial treatment",
                    ));
                }
                for (setting, factor) in run.settings.iter().zip(&request.factors) {
                    if setting.factor_id != factor.factor_id
                        || setting.unit != factor.unit
                        || setting.coded_level != 0
                        || setting.value != factor.midpoint()
                    {
                        return Err(ScreeningDesignError::new(
                            "runs.center_point.settings",
                            "center-point factors must use their declared numeric midpoints and units",
                        ));
                    }
                }
                verified_centers += 1;
            }
        }
        local_position += 1;
        current_block_run_count += 1;
        if current_block_run_count > treatment_runs_per_block + center_count {
            return Err(ScreeningDesignError::new(
                "runs.block_sequence",
                "block contains too many scheduled runs",
            ));
        }
    }

    if previous_block_id.is_none()
        || current_block_index + 1 != request.blocks.len()
        || local_position != treatment_runs_per_block + center_count
    {
        return Err(ScreeningDesignError::new(
            "runs.block_sequence",
            "one or more preregistered blocks are missing or incomplete",
        ));
    }
    if verified_treatments as usize != expected_treatment_runs
        || verified_centers as usize != expected_center_runs
        || seen_cells.len() != expected_treatment_runs
    {
        return Err(ScreeningDesignError::new(
            "runs.factorial_coverage",
            "treatment replicate cells or center-point counts are incomplete",
        ));
    }

    Ok(ScreeningDesignVerificationReceipt {
        experiment_id: request.experiment_id.clone(),
        input_snapshot_id: request.input_snapshot_id.clone(),
        algorithm_id: plan.algorithm_id.clone(),
        verified_run_count: plan.runs.len() as u32,
        verified_block_count: request.blocks.len() as u32,
        verified_factorial_combination_count: combination_count as u32,
        verified_treatment_replicate_cell_count: verified_treatments,
        verified_center_point_count: verified_centers,
        scope_note: "schedule structure verified; referenced evidence is not authenticated, and this is not power analysis, product release, field authorization, or proof of agronomic efficacy"
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(id: &str, kind: EvidenceKind) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind,
        }
    }

    fn factor(id: &str, low: f64, high: f64) -> ScreeningFactor {
        ScreeningFactor {
            factor_id: id.into(),
            label: format!("Process factor {id}"),
            unit: "degree_C".into(),
            low_value: low,
            high_value: high,
            randomization_class: FactorRandomizationClass::RandomizablePerRun,
            low_level_evidence: evidence(&format!("{id}-low-range"), EvidenceKind::Scenario),
            high_level_evidence: evidence(&format!("{id}-high-range"), EvidenceKind::Scenario),
        }
    }

    fn request() -> ScreeningDesignRequest {
        ScreeningDesignRequest {
            experiment_id: "soil-screen-001".into(),
            preregistration_id: "prereg-v1".into(),
            input_snapshot_id: "inputs-v1".into(),
            protocol_id: "bench-protocol-v1".into(),
            objective: "Screen process conditions for measured char yield and energy demand".into(),
            primary_hypothesis: "At least one declared process factor changes the primary endpoint"
                .into(),
            analysis_plan_id: "analysis-plan-v1".into(),
            randomization_seed: 0x5eed,
            factors: vec![
                factor("peak_temperature", 400.0, 500.0),
                factor("residence_time", 10.0, 30.0),
            ],
            blocks: vec![
                ExperimentBlock {
                    block_id: "day-1".into(),
                    description: "First independent operating day".into(),
                    evidence: evidence("day-1-plan", EvidenceKind::Scenario),
                },
                ExperimentBlock {
                    block_id: "day-2".into(),
                    description: "Second independent operating day".into(),
                    evidence: evidence("day-2-plan", EvidenceKind::Scenario),
                },
            ],
            replicates_per_setting_per_block: 1,
            center_point_runs_per_block: 3,
            primary_endpoint: PrimaryEndpoint {
                endpoint_id: "char-yield".into(),
                description: "Dry char mass divided by dry feedstock mass".into(),
                unit: "kg_per_kg_dry_feedstock".into(),
                measurement_method_id: "weighing-and-moisture-method-v1".into(),
                minimum_practically_meaningful_difference: 0.03,
                difference_rationale_evidence: evidence(
                    "yield-difference-rationale",
                    EvidenceKind::Literature,
                ),
            },
            bench_scale_review: BenchScaleReview {
                status: BenchScaleReviewStatus::ApprovedForThisBenchScaleProtocol,
                protocol_id: "bench-protocol-v1".into(),
                reviewed_input_snapshot_id: "inputs-v1".into(),
                review_id: "review-001".into(),
                reviewer_role: "qualified laboratory safety reviewer".into(),
                evidence: evidence("review-record-001", EvidenceKind::Measured),
            },
        }
    }

    #[test]
    fn generated_design_has_full_factorial_replication_and_center_controls() {
        let request = request();
        let plan = generate_screening_design(&request).unwrap();
        assert_eq!(plan.treatment_combination_count, 4);
        assert_eq!(plan.treatment_run_count, 8);
        assert_eq!(plan.center_point_run_count, 6);
        assert_eq!(plan.total_planned_run_count, 14);
        assert_eq!(plan.runs.len(), 14);
        assert_eq!(plan.request_snapshot, request);

        for block in &request.blocks {
            let block_runs: Vec<_> = plan
                .runs
                .iter()
                .filter(|run| run.block_id == block.block_id)
                .collect();
            assert_eq!(block_runs.len(), 7);
            assert_eq!(
                block_runs
                    .iter()
                    .filter(|run| run.kind == PlannedRunKind::CenterPointControl)
                    .count(),
                3
            );
            let treatment_rows: HashSet<_> = block_runs.iter()
                .filter_map(|run| run.standard_order)
                .collect();
            assert_eq!(treatment_rows, HashSet::from([1, 2, 3, 4]));
        }
    }

    #[test]
    fn randomization_is_reproducible_and_changes_with_seed() {
        let original = request();
        let first = generate_screening_design(&original).unwrap();
        let second = generate_screening_design(&original).unwrap();
        assert_eq!(first.runs, second.runs);

        let mut changed = original;
        changed.randomization_seed += 1;
        let third = generate_screening_design(&changed).unwrap();
        assert_ne!(
            first
                .runs
                .iter()
                .map(|r| (r.block_id.clone(), r.standard_order, r.kind))
                .collect::<Vec<_>>(),
            third
                .runs
                .iter()
                .map(|r| (r.block_id.clone(), r.standard_order, r.kind))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn factorial_settings_encode_low_high_and_center_correctly() {
        let plan = generate_screening_design(&request()).unwrap();
        let low = plan
            .runs
            .iter()
            .find(|run| {
                run.kind == PlannedRunKind::FactorialTreatment && run.standard_order == Some(1)
            })
            .unwrap();
        assert_eq!(low.settings[0].coded_level, -1);
        assert_eq!(low.settings[0].value, 400.0);
        assert_eq!(low.settings[1].coded_level, -1);
        assert_eq!(low.settings[1].value, 10.0);

        let center = plan
            .runs
            .iter()
            .find(|run| run.kind == PlannedRunKind::CenterPointControl)
            .unwrap();
        assert_eq!(center.settings[0].coded_level, 0);
        assert_eq!(center.settings[0].value, 450.0);
        assert_eq!(center.settings[1].value, 20.0);
    }

    #[test]
    fn schedule_ids_and_order_are_unique_and_contiguous() {
        let plan = generate_screening_design(&request()).unwrap();
        let ids: HashSet<_> = plan.runs.iter().map(|run| run.run_id.as_str()).collect();
        assert_eq!(ids.len(), plan.runs.len());
        assert!(plan.runs.iter().enumerate().all(|(i, run)| run.run_order == (i + 1) as u32));
    }

    #[test]
    fn invalid_factor_count_replicates_or_block_layout_are_rejected() {
        let mut input = request();
        input.factors.truncate(1);
        assert!(generate_screening_design(&input).is_err());

        input = request();
        input.replicates_per_setting_per_block = 0;
        assert!(generate_screening_design(&input).is_err());

        input = request();
        input.blocks[1].block_id = input.blocks[0].block_id.clone();
        assert!(generate_screening_design(&input).is_err());
    }

    #[test]
    fn safety_review_must_apply_to_exact_protocol_and_not_be_scenario_only() {
        let mut input = request();
        input.bench_scale_review.status = BenchScaleReviewStatus::Unknown;
        assert!(generate_screening_design(&input).is_err());

        input = request();
        input.bench_scale_review.protocol_id = "different-protocol".into();
        assert!(generate_screening_design(&input).is_err());

        input = request();
        input.bench_scale_review.evidence.kind = EvidenceKind::Scenario;
        assert!(generate_screening_design(&input).is_err());

        input = request();
        input.bench_scale_review.reviewed_input_snapshot_id = "different-inputs".into();
        assert!(generate_screening_design(&input).is_err());
    }

    #[test]
    fn center_point_controls_are_randomized_with_treatments() {
        let mut saw_center_at_start = false;
        let mut saw_treatment_at_start = false;
        let mut saw_center_at_end = false;
        let mut saw_treatment_at_end = false;

        for seed in 0..64 {
            let mut input = request();
            input.randomization_seed = seed;
            let plan = generate_screening_design(&input).unwrap();

            for block in &input.blocks {
                let block_runs: Vec<_> = plan
                    .runs
                    .iter()
                    .filter(|run| run.block_id == block.block_id)
                    .collect();
                if block_runs.first().unwrap().kind == PlannedRunKind::CenterPointControl {
                    saw_center_at_start = true;
                } else {
                    saw_treatment_at_start = true;
                }
                if block_runs.last().unwrap().kind == PlannedRunKind::CenterPointControl {
                    saw_center_at_end = true;
                } else {
                    saw_treatment_at_end = true;
                }
            }
        }

        assert!(saw_center_at_start && saw_treatment_at_start);
        assert!(saw_center_at_end && saw_treatment_at_end);
    }

    #[test]
    fn structural_verifier_accepts_valid_plan_and_rejects_corruption() {
        let generated = generate_screening_design(&request()).unwrap();
        let receipt = verify_screening_design(&generated).unwrap();
        assert_eq!(receipt.verified_run_count, 14);
        assert_eq!(receipt.verified_block_count, 2);
        assert_eq!(receipt.verified_treatment_replicate_cell_count, 8);
        assert_eq!(receipt.verified_center_point_count, 6);

        let mut tampered = generated.clone();
        let treatment = tampered.runs.iter_mut()
            .find(|run| run.kind == PlannedRunKind::FactorialTreatment)
            .unwrap();
        treatment.settings[0].value += 1.0;
        assert!(verify_screening_design(&tampered).is_err());

        let mut incomplete = generated;
        incomplete.runs.pop();
        assert!(verify_screening_design(&incomplete).is_err());
    }

    #[test]
    fn serialized_plan_roundtrip_preserves_reproducible_audit_fields() {
        let plan = generate_screening_design(&request()).unwrap();
        let encoded = serde_json::to_string(&plan).unwrap();
        let decoded: ScreeningDesignPlan = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(
            verify_screening_design(&decoded).unwrap().verified_factorial_combination_count,
            4
        );
    }

    #[test]
    fn center_points_are_optional_but_cannot_be_misconfigured() {
        let mut input = request();
        input.center_point_runs_per_block = 0;
        let plan = generate_screening_design(&input).unwrap();
        assert_eq!(plan.center_point_run_count, 0);

        input.center_point_runs_per_block = 2;
        assert!(generate_screening_design(&input).is_err());
    }

    #[test]
    fn input_rejects_nonfinite_and_reversed_factor_ranges() {
        let mut input = request();
        input.factors[0].low_value = f64::NAN;
        assert!(generate_screening_design(&input).is_err());

        input = request();
        input.factors[0].low_value = 500.0;
        input.factors[0].high_value = 400.0;
        assert!(generate_screening_design(&input).is_err());

        input = request();
        input.factors[0].randomization_class =
            FactorRandomizationClass::RestrictedOrHardToChange;
        assert!(generate_screening_design(&input).is_err());
    }
}
