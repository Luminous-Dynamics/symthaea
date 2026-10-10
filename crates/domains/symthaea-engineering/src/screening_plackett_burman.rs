// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded 12-run Plackett–Burman main-effect screening.
//!
//! This design family reduces the treatment matrix from 2^k full-factorial combinations to
//! 12 rows for 5–11 factors, where screening savings are meaningful and factor selection
//! is the goal. It is a screening proposal only: main effects can be confounded
//! with two-factor interactions, so do not use it when those interactions may be material.
//! It deliberately has its own verifier and descriptive main-effect analysis; the full-factorial
//! analysis and power formula must not be applied to these rows.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

use symthaea_agribot::soil_process::{EvidenceKind, EvidenceRef};

use crate::screening_analysis::ObservedFactorSetting;
use crate::screening_design::{
    BenchScaleReview, BenchScaleReviewStatus, ExperimentBlock, FactorSetting, PrimaryEndpoint,
    ScreeningDesignError, ScreeningFactor,
};

const ALGORITHM_ID: &str = "plackett-burman-12-run-main-effects-splitmix64-fy-centerpoints-v1";
const MIN_FACTORS: usize = 5;
const MAX_FACTORS: usize = 11;
const MIN_BLOCKS: usize = 1;
const MAX_BLOCKS: usize = 8;
const PB_RUNS: usize = 12;
const MAX_TOTAL_RUNS: usize = 1_100;

fn nonempty(value: &str, field: &'static str) -> Result<(), ScreeningDesignError> {
    if value.trim().is_empty() {
        return Err(ScreeningDesignError::new(field, "cannot be empty"));
    }
    Ok(())
}

fn evidence(value: &EvidenceRef, field: &'static str) -> Result<(), ScreeningDesignError> {
    nonempty(&value.evidence_id, field)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanDesignRequest {
    pub experiment_id: String,
    pub preregistration_id: String,
    pub input_snapshot_id: String,
    pub protocol_id: String,
    pub objective: String,
    pub primary_hypothesis: String,
    pub analysis_plan_id: String,
    pub randomization_seed: u64,
    pub factors: Vec<ScreeningFactor>,
    /// Each block contains one complete 12-run PB matrix and represents a named nuisance stratum.
    pub blocks: Vec<ExperimentBlock>,
    /// Zero, or 3–5 center-point stability controls per block.
    pub center_point_runs_per_block: u8,
    pub primary_endpoint: PrimaryEndpoint,
    pub bench_scale_review: BenchScaleReview,
}

impl PlackettBurmanDesignRequest {
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
                concat!(
                    "12-run Plackett–Burman screening is reserved for 5 to 11 quantitative ",
                    "factors; use a full factorial for smaller factor sets"
                ),
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
                "use 1 to 8 complete Plackett–Burman blocks and describe each nuisance stratum",
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
        if self.center_point_runs_per_block != 0
            && !(3..=5).contains(&self.center_point_runs_per_block)
        {
            return Err(ScreeningDesignError::new(
                "center_point_runs_per_block",
                "must be 0 or between 3 and 5",
            ));
        }
        self.primary_endpoint.validate()?;

        let review = &self.bench_scale_review;
        nonempty(&review.protocol_id, "bench_scale_review.protocol_id")?;
        nonempty(
            &review.reviewed_input_snapshot_id,
            "bench_scale_review.reviewed_input_snapshot_id",
        )?;
        if review.reviewed_input_snapshot_id != self.input_snapshot_id {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.reviewed_input_snapshot_id",
                "review must reference the exact design input snapshot",
            ));
        }
        nonempty(
            &review.reviewed_design_sha256,
            "bench_scale_review.reviewed_design_sha256",
        )?;
        if review.reviewed_design_sha256 != plackett_burman_design_sha256(self)? {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.reviewed_design_sha256",
                "review digest does not match the exact versioned Plackett–Burman design inputs",
            ));
        }
        nonempty(&review.review_id, "bench_scale_review.review_id")?;
        nonempty(&review.reviewer_role, "bench_scale_review.reviewer_role")?;
        evidence(&review.evidence, "bench_scale_review.evidence")?;
        if review.protocol_id != self.protocol_id {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.protocol_id",
                "review must apply to the exact protocol ID in this request",
            ));
        }
        if review.status != BenchScaleReviewStatus::ApprovedForThisBenchScaleProtocol {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.status",
                "schedule withheld unless this exact bench-scale protocol is declared approved",
            ));
        }
        if review.evidence.kind == EvidenceKind::Scenario {
            return Err(ScreeningDesignError::new(
                "bench_scale_review.evidence",
                "scenario evidence cannot establish a bench-scale safety review",
            ));
        }

        let per_block = PB_RUNS + self.center_point_runs_per_block as usize;
        let total = per_block.checked_mul(self.blocks.len()).ok_or_else(|| {
            ScreeningDesignError::new("run_count", "total run count overflow")
        })?;
        if total > MAX_TOTAL_RUNS {
            return Err(ScreeningDesignError::new(
                "run_count",
                "design exceeds the 1,100-run implementation bound",
            ));
        }
        Ok(total)
    }
}

#[derive(Serialize)]
struct ReviewPayload<'a> {
    schema_id: &'static str,
    algorithm_id: &'static str,
    experiment_id: &'a str,
    preregistration_id: &'a str,
    input_snapshot_id: &'a str,
    protocol_id: &'a str,
    objective: &'a str,
    primary_hypothesis: &'a str,
    analysis_plan_id: &'a str,
    randomization_seed: u64,
    factors: &'a [ScreeningFactor],
    blocks: &'a [ExperimentBlock],
    center_point_runs_per_block: u8,
    primary_endpoint: &'a PrimaryEndpoint,
}

fn sha256_hex<T: Serialize>(value: &T, field: &'static str) -> Result<String, ScreeningDesignError> {
    let bytes = serde_json::to_vec(value).map_err(|_| {
        ScreeningDesignError::new(field, "failed to serialize the versioned hash payload")
    })?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    const ALPHABET: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        encoded.push(ALPHABET[(byte >> 4) as usize] as char);
        encoded.push(ALPHABET[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
}

/// Fingerprint the exact preregistered design inputs, excluding the approval object itself.
pub fn plackett_burman_design_sha256(
    request: &PlackettBurmanDesignRequest,
) -> Result<String, ScreeningDesignError> {
    sha256_hex(
        &ReviewPayload {
            schema_id: "plackett-burman-review-payload-v1",
            algorithm_id: ALGORITHM_ID,
            experiment_id: &request.experiment_id,
            preregistration_id: &request.preregistration_id,
            input_snapshot_id: &request.input_snapshot_id,
            protocol_id: &request.protocol_id,
            objective: &request.objective,
            primary_hypothesis: &request.primary_hypothesis,
            analysis_plan_id: &request.analysis_plan_id,
            randomization_seed: request.randomization_seed,
            factors: &request.factors,
            blocks: &request.blocks,
            center_point_runs_per_block: request.center_point_runs_per_block,
            primary_endpoint: &request.primary_endpoint,
        },
        "request_sha256",
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlackettBurmanRunKind {
    MainEffectScreening,
    CenterPointControl,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanRun {
    pub run_id: String,
    pub run_order: u32,
    pub block_id: String,
    pub kind: PlackettBurmanRunKind,
    /// One-based canonical row of the 12-run PB matrix; absent for center controls.
    pub standard_order: Option<u8>,
    pub settings: Vec<FactorSetting>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanDesignPlan {
    pub algorithm_id: String,
    pub request_sha256: String,
    pub schedule_sha256: String,
    pub request_snapshot: PlackettBurmanDesignRequest,
    pub treatment_run_count: u32,
    pub center_point_run_count: u32,
    pub total_planned_run_count: u32,
    pub runs: Vec<PlackettBurmanRun>,
    pub interpretation_note: String,
}

#[derive(Serialize)]
struct SchedulePayload<'a> {
    schema_id: &'static str,
    algorithm_id: &'a str,
    request_sha256: &'a str,
    runs: &'a [PlackettBurmanRun],
}

fn schedule_sha256(
    algorithm_id: &str,
    request_sha256: &str,
    runs: &[PlackettBurmanRun],
) -> Result<String, ScreeningDesignError> {
    sha256_hex(
        &SchedulePayload {
            schema_id: "plackett-burman-schedule-payload-v1",
            algorithm_id,
            request_sha256,
            runs,
        },
        "schedule_sha256",
    )
}

/// Canonical NIST 12-run, up-to-11-factor Plackett–Burman matrix.
/// Row 1 is all +1; rows 2–12 are cyclic right rotations of the published base row.
fn canonical_pb12_matrix() -> Vec<Vec<i8>> {
    let mut rows = Vec::with_capacity(PB_RUNS);
    rows.push(vec![1; MAX_FACTORS]);
    let mut row = vec![-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1];
    for _ in 0..11 {
        rows.push(row.clone());
        row.rotate_right(1);
    }
    rows
}


/// Independently reconstruct the published 12 × 11 sign matrix for verification.
/// This uses the table's textual base pattern and column indexing rather than the generator's
/// vector rotation implementation, so a generator-matrix drift is not silently self-verified.
fn verifier_pb12_matrix() -> Vec<Vec<i8>> {
    let base: Vec<i8> = "-+-+++---+-"
        .bytes()
        .map(|sign| if sign == b'+' { 1 } else { -1 })
        .collect();
    let mut rows = vec![vec![1; MAX_FACTORS]];
    for shift in 0..11 {
        rows.push(
            (0..MAX_FACTORS)
                .map(|column| base[(column + MAX_FACTORS - shift) % MAX_FACTORS])
                .collect(),
        );
    }
    rows
}

fn center_point_positions(treatment_count: usize, center_count: usize) -> HashSet<usize> {
    let total = treatment_count + center_count;
    if center_count == 0 {
        return HashSet::new();
    }
    (0..center_count)
        .map(|index| index * (total - 1) / (center_count - 1))
        .collect()
}

fn next_random(state: &mut u64) -> u64 {
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

fn settings_for_row(
    factors: &[ScreeningFactor],
    matrix: &[Vec<i8>],
    row: usize,
    center: bool,
) -> Vec<FactorSetting> {
    factors
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            let coded_level = if center { 0 } else { matrix[row][index] };
            let value = match coded_level {
                -1 => factor.low_value,
                1 => factor.high_value,
                _ => factor.low_value / 2.0 + factor.high_value / 2.0,
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

/// Generate a randomized, blocked, 12-run Plackett–Burman screen for 5–11 numeric factors.
/// This is for main-effect screening under an explicit negligible-interactions assumption;
/// it is not a fractional-factorial interaction model or proof of adequate power.
pub fn generate_plackett_burman_design(
    request: &PlackettBurmanDesignRequest,
) -> Result<PlackettBurmanDesignPlan, ScreeningDesignError> {
    let total_runs = request.validate()?;
    let request_sha256 = plackett_burman_design_sha256(request)?;
    let center_count = request.center_point_runs_per_block as usize;
    let total_per_block = PB_RUNS + center_count;
    let centers = center_point_positions(PB_RUNS, center_count);
    let matrix = canonical_pb12_matrix();
    let mut rng_state = request.randomization_seed;
    let mut runs = Vec::with_capacity(total_runs);
    let mut next_order = 1_u32;
    let mut treatment_run_count = 0_u32;
    let mut center_point_run_count = 0_u32;

    for block in &request.blocks {
        let mut rows: Vec<usize> = (0..PB_RUNS).collect();
        shuffle(&mut rows, &mut rng_state);
        let mut treatment_cursor = 0;
        for position in 0..total_per_block {
            let (kind, standard_order, settings) = if centers.contains(&position) {
                center_point_run_count += 1;
                (
                    PlackettBurmanRunKind::CenterPointControl,
                    None,
                    settings_for_row(&request.factors, &matrix, 0, true),
                )
            } else {
                let row = rows[treatment_cursor];
                treatment_cursor += 1;
                treatment_run_count += 1;
                (
                    PlackettBurmanRunKind::MainEffectScreening,
                    Some((row + 1) as u8),
                    settings_for_row(&request.factors, &matrix, row, false),
                )
            };
            let run_id = format!(
                "{}-{}-{:04}",
                request.experiment_id, block.block_id, next_order
            );
            runs.push(PlackettBurmanRun {
                run_id,
                run_order: next_order,
                block_id: block.block_id.clone(),
                kind,
                standard_order,
                settings,
            });
            next_order += 1;
        }
    }

    let schedule_sha256 = schedule_sha256(&ALGORITHM_ID, &request_sha256, &runs)?;
    Ok(PlackettBurmanDesignPlan {
        algorithm_id: ALGORITHM_ID.into(),
        request_sha256,
        schedule_sha256,
        request_snapshot: request.clone(),
        treatment_run_count,
        center_point_run_count,
        total_planned_run_count: runs.len() as u32,
        runs,
        interpretation_note: concat!(
            "12-run Plackett–Burman main-effect screening proposal only. Main effects are generally ",
            "confounded with two-factor interactions; use only when that limitation is defensible, ",
            "then confirm selected factors in a follow-up design. No exact power analysis, ",
            "interaction estimates, safety release, field authorization, or agronomic claim is provided."
        )
        .into(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanVerificationReceipt {
    pub experiment_id: String,
    pub request_sha256: String,
    pub schedule_sha256: String,
    pub verified_run_count: u32,
    pub verified_block_count: u32,
    pub verified_factor_count: u32,
    pub verified_treatment_rows: u32,
    pub verified_center_points: u32,
    pub scope_note: String,
}

/// Rebuild the seeded assignment order in a distinct code path, not by invoking generator
/// shuffle helpers. The seed is reproducible but is not a cryptographic trust anchor.
fn verifier_expected_orders(request: &PlackettBurmanDesignRequest) -> Vec<Vec<usize>> {
    let mut state = request.randomization_seed;
    let mut result = Vec::with_capacity(request.blocks.len());
    for _ in &request.blocks {
        let mut rows: Vec<usize> = (0..PB_RUNS).collect();
        for index in (1..rows.len()).rev() {
            let bound = (index + 1) as u64;
            let threshold = bound.wrapping_neg() % bound;
            let other = loop {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                let value = z ^ (z >> 31);
                if value >= threshold {
                    break (value % bound) as usize;
                }
            };
            rows.swap(index, other as usize);
        }
        result.push(rows);
    }
    result
}

/// Independently recheck fingerprints, the seed-derived order, complete PB rows and settings.
pub fn verify_plackett_burman_design(
    plan: &PlackettBurmanDesignPlan,
) -> Result<PlackettBurmanVerificationReceipt, ScreeningDesignError> {
    let request = &plan.request_snapshot;
    let expected_total = request.validate()?;
    let request_sha = plackett_burman_design_sha256(request)?;
    if request_sha != plan.request_sha256 {
        return Err(ScreeningDesignError::new(
            "plan.request_sha256",
            "stored request fingerprint does not match the approved design inputs",
        ));
    }
    if schedule_sha256(&plan.algorithm_id, &request_sha, &plan.runs)? != plan.schedule_sha256 {
        return Err(ScreeningDesignError::new(
            "plan.schedule_sha256",
            "stored schedule checksum does not match the exact run artifact",
        ));
    }
    if plan.algorithm_id != ALGORITHM_ID {
        return Err(ScreeningDesignError::new(
            "algorithm_id",
            "unknown or mismatched Plackett–Burman algorithm ID",
        ));
    }
    let expected_treatment = PB_RUNS * request.blocks.len();
    let expected_centers = request.center_point_runs_per_block as usize * request.blocks.len();
    if plan.runs.len() != expected_total
        || plan.total_planned_run_count as usize != expected_total
        || plan.treatment_run_count as usize != expected_treatment
        || plan.center_point_run_count as usize != expected_centers
    {
        return Err(ScreeningDesignError::new(
            "plan.run_counts",
            "stored run counts do not match the preregistered PB design",
        ));
    }

    let expected_orders = verifier_expected_orders(request);
    let matrix = verifier_pb12_matrix();
    let center_count = request.center_point_runs_per_block as usize;
    let expected_centers_positions = center_point_positions(PB_RUNS, center_count);
    let per_block = PB_RUNS + center_count;
    let mut ids = HashSet::new();
    let mut rows_seen: Vec<HashSet<u8>> = vec![HashSet::new(); request.blocks.len()];
    let mut treatment_positions = vec![0_usize; request.blocks.len()];
    let mut current_block: Option<usize> = None;
    let mut local_position = 0_usize;
    let mut verified_treatments = 0_u32;
    let mut verified_centers = 0_u32;

    for (index, run) in plan.runs.iter().enumerate() {
        if run.run_order != (index + 1) as u32 {
            return Err(ScreeningDesignError::new(
                "runs.run_order",
                "run order must be contiguous and one-based",
            ));
        }
        let expected_id = format!(
            "{}-{}-{:04}",
            request.experiment_id, run.block_id, run.run_order
        );
        if run.run_id != expected_id || !ids.insert(run.run_id.as_str()) {
            return Err(ScreeningDesignError::new(
                "runs.run_id",
                "run IDs must be unique and match the declared naming scheme",
            ));
        }
        let block_index = request
            .blocks
            .iter()
            .position(|block| block.block_id == run.block_id)
            .ok_or_else(|| {
                ScreeningDesignError::new("runs.block_id", "run references an unknown block")
            })?;
        if current_block != Some(block_index) {
            if let Some(previous) = current_block {
                if local_position != per_block || block_index != previous + 1 {
                    return Err(ScreeningDesignError::new(
                        "runs.block_sequence",
                        "blocks must be complete, contiguous, and in preregistered order",
                    ));
                }
            } else if block_index != 0 {
                return Err(ScreeningDesignError::new(
                    "runs.block_sequence",
                    "first scheduled block must be the first preregistered block",
                ));
            }
            current_block = Some(block_index);
            local_position = 0;
        }
        if run.settings.len() != request.factors.len() {
            return Err(ScreeningDesignError::new(
                "runs.settings",
                "each run must carry one setting per factor",
            ));
        }

        match run.kind {
            PlackettBurmanRunKind::CenterPointControl => {
                if !expected_centers_positions.contains(&local_position)
                    || run.standard_order.is_some()
                {
                    return Err(ScreeningDesignError::new(
                        "runs.center_point",
                        "center controls must occupy preregistered positions and have no PB row",
                    ));
                }
                for (setting, factor) in run.settings.iter().zip(&request.factors) {
                    let midpoint = factor.low_value / 2.0 + factor.high_value / 2.0;
                    if setting.factor_id != factor.factor_id
                        || setting.unit != factor.unit
                        || setting.coded_level != 0
                        || setting.value != midpoint
                    {
                        return Err(ScreeningDesignError::new(
                            "runs.center_point.settings",
                            "center control settings must match the declared numeric midpoints",
                        ));
                    }
                }
                verified_centers += 1;
            }
            PlackettBurmanRunKind::MainEffectScreening => {
                if expected_centers_positions.contains(&local_position) {
                    return Err(ScreeningDesignError::new(
                        "runs.center_point_position",
                        "a center point is missing from its fixed position",
                    ));
                }
                let row = run.standard_order.ok_or_else(|| {
                    ScreeningDesignError::new("runs.standard_order", "PB treatment lacks its row")
                })? as usize;
                if row == 0 || row > PB_RUNS {
                    return Err(ScreeningDesignError::new(
                        "runs.standard_order",
                        "PB row must be within 1..=12",
                    ));
                }
                let treatment_position = treatment_positions[block_index];
                if expected_orders[block_index].get(treatment_position).copied() != Some(row - 1) {
                    return Err(ScreeningDesignError::new(
                        "runs.randomization",
                        "treatment order does not match the declared seed and PB algorithm",
                    ));
                }
                treatment_positions[block_index] += 1;
                if !rows_seen[block_index].insert(row as u8) {
                    return Err(ScreeningDesignError::new(
                        "runs.pb_coverage",
                        "each complete PB block must contain every canonical row exactly once",
                    ));
                }
                for (factor_index, (setting, factor)) in
                    run.settings.iter().zip(&request.factors).enumerate()
                {
                    let coded = matrix[row - 1][factor_index];
                    let value = if coded < 0 { factor.low_value } else { factor.high_value };
                    if setting.factor_id != factor.factor_id
                        || setting.unit != factor.unit
                        || setting.coded_level != coded
                        || setting.value != value
                    {
                        return Err(ScreeningDesignError::new(
                            "runs.settings",
                            "factor ID, unit, level or numeric value disagrees with the canonical PB row",
                        ));
                    }
                }
                verified_treatments += 1;
            }
        }
        local_position += 1;
    }

    if current_block.is_none()
        || local_position != per_block
        || request.blocks.len() != current_block.unwrap_or(0) + 1
        || rows_seen.iter().any(|rows| rows.len() != PB_RUNS)
    {
        return Err(ScreeningDesignError::new(
            "runs.block_sequence",
            "one or more PB blocks are absent or incomplete",
        ));
    }

    Ok(PlackettBurmanVerificationReceipt {
        experiment_id: request.experiment_id.clone(),
        request_sha256: request_sha,
        schedule_sha256: plan.schedule_sha256.clone(),
        verified_run_count: plan.runs.len() as u32,
        verified_block_count: request.blocks.len() as u32,
        verified_factor_count: request.factors.len() as u32,
        verified_treatment_rows: verified_treatments,
        verified_center_points: verified_centers,
        scope_note: concat!(
            "PB schedule structure and declared seed order verified; interactions remain aliased ",
            "and external evidence/approval identities are not authenticated"
        )
        .into(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanMainEffect {
    pub factor_id: String,
    /// Descriptive high-level mean minus low-level mean; not a p-value or causal guarantee.
    pub effect_estimate: f64,
    pub unit: String,
    pub treatment_observation_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanCenterPointDiagnostic {
    pub block_id: String,
    pub treatment_mean: f64,
    pub center_point_mean: f64,
    pub center_point_minus_treatment_mean: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanResponseObservation {
    pub observation_id: String,
    pub run_id: String,
    pub request_sha256: String,
    pub schedule_sha256: String,
    pub actual_factor_settings: Vec<ObservedFactorSetting>,
    pub endpoint_id: String,
    pub outcome_value: f64,
    pub outcome_unit: String,
    pub measurement_method_id: String,
    pub evidence: EvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlackettBurmanResponseAnalysis {
    pub algorithm_id: String,
    pub experiment_id: String,
    pub preregistration_id: String,
    pub input_snapshot_id: String,
    pub analysis_plan_id: String,
    pub endpoint_id: String,
    pub endpoint_unit: String,
    pub design_verification: PlackettBurmanVerificationReceipt,
    pub request_snapshot: PlackettBurmanDesignRequest,
    pub schedule_snapshot: Vec<PlackettBurmanRun>,
    pub observations_in_run_order: Vec<PlackettBurmanResponseObservation>,
    pub treatment_observation_count: u32,
    pub center_point_observation_count: u32,
    pub main_effects: Vec<PlackettBurmanMainEffect>,
    pub block_center_point_diagnostics: Vec<PlackettBurmanCenterPointDiagnostic>,
    pub treatment_mean: f64,
    pub scope_note: String,
}

/// Descriptive main-effect contrasts for a complete PB response set.
/// No interaction effects, p-values, confidence intervals or causal claims are generated.
pub fn analyze_plackett_burman_responses(
    plan: &PlackettBurmanDesignPlan,
    observations: &[PlackettBurmanResponseObservation],
) -> Result<PlackettBurmanResponseAnalysis, ScreeningDesignError> {
    let verified = verify_plackett_burman_design(plan)?;
    if observations.len() != plan.runs.len() {
        return Err(ScreeningDesignError::new(
            "observations",
            "exactly one measured primary-endpoint outcome is required for every planned run",
        ));
    }
    let request = &plan.request_snapshot;
    let endpoint = &request.primary_endpoint;
    let planned_by_id: HashMap<&str, &PlackettBurmanRun> =
        plan.runs.iter().map(|run| (run.run_id.as_str(), run)).collect();
    let mut observation_ids = HashSet::new();
    let mut observations_by_run: HashMap<&str, &PlackettBurmanResponseObservation> = HashMap::new();

    for observation in observations {
        if observation.observation_id.trim().is_empty()
            || !observation_ids.insert(observation.observation_id.as_str())
        {
            return Err(ScreeningDesignError::new(
                "observations.observation_id",
                "observation IDs must be non-empty and unique",
            ));
        }
        if observation.request_sha256 != plan.request_sha256
            || observation.schedule_sha256 != plan.schedule_sha256
        {
            return Err(ScreeningDesignError::new(
                "observations.fingerprint",
                "observation is bound to a different preregistered design or exact run schedule",
            ));
        }
        let run = planned_by_id.get(observation.run_id.as_str()).ok_or_else(|| {
            ScreeningDesignError::new("observations.run_id", "observation references an unknown run")
        })?;
        if observation.actual_factor_settings.len() != request.factors.len() {
            return Err(ScreeningDesignError::new(
                "observations.actual_factor_settings",
                "one achieved setting is required for every preregistered factor",
            ));
        }
        let mut seen_factor_ids = HashSet::new();
        for actual in &observation.actual_factor_settings {
            if actual.factor_id.trim().is_empty() || !seen_factor_ids.insert(actual.factor_id.as_str()) {
                return Err(ScreeningDesignError::new(
                    "observations.actual_factor_settings.factor_id",
                    "factor identifiers must be non-empty and unique",
                ));
            }
            let factor = request.factors.iter().find(|f| f.factor_id == actual.factor_id).ok_or_else(|| {
                ScreeningDesignError::new("observations.actual_factor_settings.factor_id", "unknown factor")
            })?;
            let assigned = run.settings.iter().find(|s| s.factor_id == actual.factor_id).ok_or_else(|| {
                ScreeningDesignError::new(
                    "observations.actual_factor_settings.factor_id",
                    "factor missing from planned run",
                )
            })?;
            if !actual.value.is_finite() || actual.unit != factor.unit {
                return Err(ScreeningDesignError::new(
                    "observations.actual_factor_settings",
                    "achieved value must be finite and use the preregistered unit",
                ));
            }
            if actual.evidence.kind != EvidenceKind::Measured || actual.evidence.evidence_id.trim().is_empty() {
                return Err(ScreeningDesignError::new(
                    "observations.actual_factor_settings.evidence",
                    "achieved settings require non-empty measured evidence",
                ));
            }
            let deviation = (actual.value - assigned.value).abs();
            if !deviation.is_finite() || deviation > factor.max_absolute_deviation {
                return Err(ScreeningDesignError::new(
                    "observations.actual_factor_settings.value",
                    "achieved setting lies outside the preregistered tolerance",
                ));
            }
        }
        if seen_factor_ids.len() != request.factors.len() {
            return Err(ScreeningDesignError::new(
                "observations.actual_factor_settings",
                "one or more achieved factor settings are missing",
            ));
        }
        if observation.endpoint_id != endpoint.endpoint_id
            || observation.outcome_unit != endpoint.unit
            || observation.measurement_method_id != endpoint.measurement_method_id
        {
            return Err(ScreeningDesignError::new(
                "observations.endpoint",
                "endpoint, unit and method must match the preregistration",
            ));
        }
        if !observation.outcome_value.is_finite()
            || observation.evidence.kind != EvidenceKind::Measured
            || observation.evidence.evidence_id.trim().is_empty()
        {
            return Err(ScreeningDesignError::new(
                "observations.evidence",
                "outcomes must be finite measured observations with evidence IDs",
            ));
        }
        if observations_by_run.insert(observation.run_id.as_str(), observation).is_some() {
            return Err(ScreeningDesignError::new(
                "observations.run_id",
                "exactly one observation is allowed per planned run",
            ));
        }
    }
    if observations_by_run.len() != planned_by_id.len() {
        return Err(ScreeningDesignError::new(
            "observations",
            "one or more planned runs have no measured response",
        ));
    }

    let mut ordered = Vec::with_capacity(plan.runs.len());
    for run in &plan.runs {
        ordered.push((*observations_by_run.get(run.run_id.as_str()).ok_or_else(|| {
            ScreeningDesignError::new("observations", "planned run lacks a response")
        })?).clone());
    }

    let matrix = canonical_pb12_matrix();
    let treatment_count = (PB_RUNS * request.blocks.len()) as u32;
    let mut effect_sums = vec![0.0_f64; request.factors.len()];
    let mut treatment_sum_by_block: HashMap<&str, f64> = HashMap::new();
    let mut treatment_count_by_block: HashMap<&str, u32> = HashMap::new();
    let mut center_sum_by_block: HashMap<&str, f64> = HashMap::new();
    let mut center_count_by_block: HashMap<&str, u32> = HashMap::new();
    let mut treatment_sum = 0.0_f64;
    let mut centers = 0_u32;
    for run in &plan.runs {
        let outcome = observations_by_run[run.run_id.as_str()].outcome_value;
        match run.kind {
            PlackettBurmanRunKind::MainEffectScreening => {
                let row = run.standard_order.ok_or_else(|| {
                    ScreeningDesignError::new("runs.standard_order", "main-effect run lacks PB row")
                })? as usize - 1;
                for (factor_index, sum) in effect_sums.iter_mut().enumerate() {
                    *sum += matrix[row][factor_index] as f64 * outcome;
                    if !sum.is_finite() {
                        return Err(ScreeningDesignError::new(
                            "observations.outcome_value",
                            "main-effect contrast overflow",
                        ));
                    }
                }
                let block_sum = treatment_sum_by_block.entry(run.block_id.as_str()).or_insert(0.0);
                *block_sum += outcome;
                *treatment_count_by_block.entry(run.block_id.as_str()).or_insert(0) += 1;
                treatment_sum += outcome;
                if !block_sum.is_finite() || !treatment_sum.is_finite() {
                    return Err(ScreeningDesignError::new("observations.outcome_value", "treatment sum overflow"));
                }
            }
            PlackettBurmanRunKind::CenterPointControl => {
                let block_sum = center_sum_by_block.entry(run.block_id.as_str()).or_insert(0.0);
                *block_sum += outcome;
                *center_count_by_block.entry(run.block_id.as_str()).or_insert(0) += 1;
                if !block_sum.is_finite() {
                    return Err(ScreeningDesignError::new("observations.outcome_value", "center-point sum overflow"));
                }
                centers += 1;
            }
        }
    }
    let mut main_effects = Vec::with_capacity(request.factors.len());
    for (factor_index, factor) in request.factors.iter().enumerate() {
        let effect = 2.0 * effect_sums[factor_index] / treatment_count as f64;
        if !effect.is_finite() {
            return Err(ScreeningDesignError::new("main_effects", "effect estimate is non-finite"));
        }
        main_effects.push(PlackettBurmanMainEffect {
            factor_id: factor.factor_id.clone(),
            effect_estimate: effect,
            unit: endpoint.unit.clone(),
            treatment_observation_count: treatment_count,
        });
    }
    let treatment_mean = treatment_sum / treatment_count as f64;
    if !treatment_mean.is_finite() {
        return Err(ScreeningDesignError::new("treatment_mean", "mean is non-finite"));
    }
    let mut diagnostics = Vec::new();
    if request.center_point_runs_per_block > 0 {
        for block in &request.blocks {
            let block_treatment_count = *treatment_count_by_block.get(block.block_id.as_str()).unwrap_or(&0);
            let block_center_count = *center_count_by_block.get(block.block_id.as_str()).unwrap_or(&0);
            if block_treatment_count != PB_RUNS as u32
                || block_center_count != request.center_point_runs_per_block as u32
            {
                return Err(ScreeningDesignError::new(
                    "observations.center_points",
                    "block must contain all 12 PB treatments and the declared center-point count",
                ));
            }
            let block_treatment_mean =
                treatment_sum_by_block[block.block_id.as_str()] / block_treatment_count as f64;
            let block_center_mean =
                center_sum_by_block[block.block_id.as_str()] / block_center_count as f64;
            diagnostics.push(PlackettBurmanCenterPointDiagnostic {
                block_id: block.block_id.clone(),
                treatment_mean: block_treatment_mean,
                center_point_mean: block_center_mean,
                center_point_minus_treatment_mean: block_center_mean - block_treatment_mean,
                unit: endpoint.unit.clone(),
            });
        }
    }

    Ok(PlackettBurmanResponseAnalysis {
        algorithm_id: ALGORITHM_ID.into(),
        experiment_id: request.experiment_id.clone(),
        preregistration_id: request.preregistration_id.clone(),
        input_snapshot_id: request.input_snapshot_id.clone(),
        analysis_plan_id: request.analysis_plan_id.clone(),
        endpoint_id: endpoint.endpoint_id.clone(),
        endpoint_unit: endpoint.unit.clone(),
        design_verification: verified,
        request_snapshot: request.clone(),
        schedule_snapshot: plan.runs.clone(),
        observations_in_run_order: ordered,
        treatment_observation_count: treatment_count,
        center_point_observation_count: centers,
        main_effects,
        block_center_point_diagnostics: diagnostics,
        treatment_mean,
        scope_note: concat!(
            "descriptive PB high-minus-low main effects only; these estimates can be biased by ",
            "aliased two-factor interactions. No inferential claims, p-values, confidence intervals, ",
            "safety approval, field authorization, or proof of agronomic efficacy."
        )
        .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening_design::FactorRandomizationClass;

    fn ev(id: &str, kind: EvidenceKind) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind,
        }
    }

    fn factor(index: usize) -> ScreeningFactor {
        let id = format!("factor_{index:02}");
        ScreeningFactor {
            factor_id: id.clone(),
            label: format!("Process factor {index}"),
            unit: "unit".into(),
            low_value: 10.0,
            high_value: 20.0 + index as f64,
            max_absolute_deviation: 1.0,
            deviation_tolerance_evidence: ev(
                &format!("{id}-tol"),
                EvidenceKind::Literature,
            ),
            randomization_class: FactorRandomizationClass::RandomizablePerRun,
            low_level_evidence: ev(&format!("{id}-low"), EvidenceKind::Literature),
            high_level_evidence: ev(&format!("{id}-high"), EvidenceKind::Literature),
        }
    }

    fn request(factor_count: usize, centers: u8) -> PlackettBurmanDesignRequest {
        let mut request = PlackettBurmanDesignRequest {
            experiment_id: "pb-soil-001".into(),
            preregistration_id: "prereg-pb-v1".into(),
            input_snapshot_id: "pb-input-v1".into(),
            protocol_id: "pb-bench-v1".into(),
            objective: "Screen process factors before selecting a follow-up design".into(),
            primary_hypothesis: "At least one main effect is practically meaningful".into(),
            analysis_plan_id: "pb-analysis-v1".into(),
            randomization_seed: 42,
            factors: (0..factor_count).map(factor).collect(),
            blocks: vec![
                ExperimentBlock {
                    block_id: "day-1".into(),
                    description: "First independent bench block".into(),
                    evidence: ev("day-1-record", EvidenceKind::Literature),
                },
                ExperimentBlock {
                    block_id: "day-2".into(),
                    description: "Second independent bench block".into(),
                    evidence: ev("day-2-record", EvidenceKind::Literature),
                },
            ],
            center_point_runs_per_block: centers,
            primary_endpoint: PrimaryEndpoint {
                endpoint_id: "yield".into(),
                description: "Dry-basis yield".into(),
                unit: "kg_per_kg".into(),
                measurement_method_id: "mass-method-v1".into(),
                minimum_practically_meaningful_difference: 0.03,
                difference_rationale_evidence: ev(
                    "difference-rationale",
                    EvidenceKind::Literature,
                ),
            },
            bench_scale_review: BenchScaleReview {
                status: BenchScaleReviewStatus::ApprovedForThisBenchScaleProtocol,
                protocol_id: "pb-bench-v1".into(),
                reviewed_input_snapshot_id: "pb-input-v1".into(),
                reviewed_design_sha256: String::new(),
                review_id: "pb-review-001".into(),
                reviewer_role: "qualified bench-scale reviewer".into(),
                evidence: ev("pb-review-record", EvidenceKind::Measured),
            },
        };
        let digest = plackett_burman_design_sha256(&request).unwrap();
        request.bench_scale_review.reviewed_design_sha256 = digest;
        request
    }

    fn observations_for(
        plan: &PlackettBurmanDesignPlan,
        scenario_outcomes: bool,
    ) -> Vec<PlackettBurmanResponseObservation> {
        let matrix = canonical_pb12_matrix();
        plan.runs
            .iter()
            .map(|run| {
                let actual_factor_settings = run
                    .settings
                    .iter()
                    .map(|setting| ObservedFactorSetting {
                        factor_id: setting.factor_id.clone(),
                        value: setting.value,
                        unit: setting.unit.clone(),
                        evidence: ev(
                            &format!("achieved-{}", run.run_id),
                            EvidenceKind::Measured,
                        ),
                    })
                    .collect();
                let block_index = plan
                    .request_snapshot
                    .blocks
                    .iter()
                    .position(|block| block.block_id == run.block_id)
                    .unwrap_or(0);
                let outcome_value = match run.kind {
                    PlackettBurmanRunKind::CenterPointControl => {
                        100.0 + 13.0 * block_index as f64
                    }
                    PlackettBurmanRunKind::MainEffectScreening => {
                        let row = run.standard_order.unwrap() as usize - 1;
                        100.0
                            + 13.0 * block_index as f64
                            + run
                                .settings
                                .iter()
                                .enumerate()
                                .map(|(index, _)| 2.5 * matrix[row][index] as f64)
                                .sum::<f64>()
                    }
                };
                PlackettBurmanResponseObservation {
                    observation_id: format!("obs-{}", run.run_id),
                    run_id: run.run_id.clone(),
                    request_sha256: plan.request_sha256.clone(),
                    schedule_sha256: plan.schedule_sha256.clone(),
                    actual_factor_settings,
                    endpoint_id: "yield".into(),
                    outcome_value,
                    outcome_unit: "kg_per_kg".into(),
                    measurement_method_id: "mass-method-v1".into(),
                    evidence: ev(
                        &format!("outcome-{}", run.run_id),
                        if scenario_outcomes {
                            EvidenceKind::Scenario
                        } else {
                            EvidenceKind::Measured
                        },
                    ),
                }
            })
            .collect()
    }

    #[test]
    fn canonical_matrix_is_balanced_and_pairwise_orthogonal() {
        let matrix = canonical_pb12_matrix();
        assert_eq!(matrix.len(), 12);
        assert!(matrix.iter().all(|row| row.len() == 11));
        for first in 0..11 {
            assert_eq!(
                matrix.iter().map(|row| row[first] as i32).sum::<i32>(),
                0
            );
            for second in (first + 1)..11 {
                let dot = matrix
                    .iter()
                    .map(|row| row[first] as i32 * row[second] as i32)
                    .sum::<i32>();
                assert_eq!(
                    dot, 0,
                    "columns {first} and {second} are not orthogonal"
                );
            }
        }
    }

    #[test]
    fn canonical_matrix_matches_independent_verifier_matrix() {
        assert_eq!(canonical_pb12_matrix(), verifier_pb12_matrix());
    }

    #[test]
    fn full_factorial_run_growth_is_reduced_by_bounded_pb_matrix() {
        let small = generate_plackett_burman_design(&request(5, 0)).unwrap();
        let large = generate_plackett_burman_design(&request(11, 0)).unwrap();
        assert_eq!(small.treatment_run_count, 24);
        assert_eq!(large.treatment_run_count, 24);
        assert_eq!(large.total_planned_run_count, 24);
        assert_eq!(1_usize << 5, 32);
        assert_eq!(1_usize << 11, 2_048);
        assert!(verify_plackett_burman_design(&large).is_ok());

        let with_centers = generate_plackett_burman_design(&request(11, 5)).unwrap();
        assert_eq!(with_centers.total_planned_run_count, 34);
        assert!(verify_plackett_burman_design(&with_centers).is_ok());
    }

    #[test]
    fn generation_is_deterministic_and_seed_bound() {
        let input = request(7, 3);
        let first = generate_plackett_burman_design(&input).unwrap();
        let second = generate_plackett_burman_design(&input).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            verify_plackett_burman_design(&first)
                .unwrap()
                .verified_treatment_rows,
            24
        );

        let mut changed = input;
        changed.randomization_seed += 1;
        let changed_digest = plackett_burman_design_sha256(&changed).unwrap();
        changed.bench_scale_review.reviewed_design_sha256 = changed_digest;
        let other = generate_plackett_burman_design(&changed).unwrap();
        assert_ne!(first.schedule_sha256, other.schedule_sha256);
    }

    #[test]
    fn checksum_recalculation_does_not_hide_seed_order_tampering() {
        let input = request(5, 0);
        let mut plan = generate_plackett_burman_design(&input).unwrap();
        let positions = plan
            .runs
            .iter()
            .enumerate()
            .filter(|(_, run)| {
                run.block_id == "day-1"
                    && run.kind == PlackettBurmanRunKind::MainEffectScreening
            })
            .map(|(index, _)| index)
            .take(2)
            .collect::<Vec<_>>();
        let (first, second) = (positions[0], positions[1]);
        let first_row = plan.runs[first].standard_order;
        let first_settings = plan.runs[first].settings.clone();

        plan.runs[first].standard_order = plan.runs[second].standard_order;
        plan.runs[first].settings = plan.runs[second].settings.clone();
        plan.runs[second].standard_order = first_row;
        plan.runs[second].settings = first_settings;
        plan.schedule_sha256 = schedule_sha256(
            &plan.algorithm_id,
            &plan.request_sha256,
            &plan.runs,
        )
        .unwrap();

        let error = verify_plackett_burman_design(&plan).unwrap_err();
        assert_eq!(error.field, "runs.randomization");
    }

    #[test]
    fn descriptive_analysis_recovers_additive_main_effects() {
        let plan = generate_plackett_burman_design(&request(5, 0)).unwrap();
        let observations = observations_for(&plan, false);
        let result = analyze_plackett_burman_responses(&plan, &observations).unwrap();

        assert_eq!(result.main_effects.len(), 5);
        for effect in result.main_effects {
            assert!(
                (effect.effect_estimate - 5.0).abs() < 1e-9,
                "{}",
                effect.factor_id
            );
        }
    }

    #[test]
    fn smaller_factor_sets_should_use_full_factorial() {
        assert!(generate_plackett_burman_design(&request(4, 0)).is_err());
    }

    #[test]
    fn missing_and_scenario_responses_fail_closed() {
        let plan = generate_plackett_burman_design(&request(5, 0)).unwrap();
        assert!(analyze_plackett_burman_responses(&plan, &[]).is_err());

        let scenario_observations = observations_for(&plan, true);
        assert!(
            analyze_plackett_burman_responses(&plan, &scenario_observations).is_err()
        );

        let mut unapproved = request(5, 0);
        unapproved.bench_scale_review.status = BenchScaleReviewStatus::Unknown;
        let unapproved_digest = plackett_burman_design_sha256(&unapproved).unwrap();
        unapproved.bench_scale_review.reviewed_design_sha256 = unapproved_digest;
        assert!(generate_plackett_burman_design(&unapproved).is_err());
    }
}
