// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic candidate generation for a synthetic road-crossing scenario.
//!
//! This module owns scenario *design*, not execution or vehicle physics. It
//! provides reproducible stratified and seeded-pseudorandom candidate sets for Symtropy.
//! Every range is a synthetic design assumption, not an empirical distribution
//! or a claim about real-world frequency.

use blake3::Hasher;
use serde::{Deserialize, Serialize};

pub const INTERSECTION_CANDIDATE_SCHEMA: &str =
    "symthaea.vehicle.intersection-candidate.v0.1";
const STRATIFIED_GENERATOR_POLICY_ID: &str = "stratified-affine-lhs-v0.1";
const RANDOM_GENERATOR_POLICY_ID: &str = "seeded-splitmix64-random-v0.1";
const COUNTEREXAMPLE_MINIMIZATION_POLICY_ID: &str = "counterexample-coordinate-minimize-v0.1";
pub const MAX_COUNTEREXAMPLE_MINIMIZATION_EVALUATIONS: u32 = 256;
const MAX_COORDINATE_SHRINK_STEPS: u32 = 12;
pub const MAX_INTERSECTION_CANDIDATES: u32 = 4096;

const ROAD_GEOMETRY_ID: &str = "two-approach-one-marked-crossing-synthetic-v0.1";
const EGO_SPAWN_X_M: f64 = -40.0;
const EGO_LANE_CENTER_Y_M: f64 = 0.0;
const ACTOR_SPAWN_X_M: f64 = 0.0;
const ACTOR_SPAWN_Y_M: f64 = -6.0;
const ACTOR_CROSSING_SPEED_MPS: f64 = 1.4;
const ACTOR_CROSSING_PATH_LENGTH_M: f64 = 12.0;
const ACTOR_CROSSING_DIRECTION_Y: i8 = 1;
const SIMULATION_TIMESTEP_S: f64 = 0.02;
const SIMULATION_HORIZON_S: f64 = 8.0;

const EGO_SPEED_RANGE_MPS: (f64, f64) = (6.0, 22.0);
const CROSSING_START_RANGE_S: (f64, f64) = (0.8, 4.0);
const OCCLUSION_RANGE: (f64, f64) = (0.0, 0.9);
const SENSOR_DELAY_RANGE_S: (f64, f64) = (0.0, 0.25);
const SENSOR_DROPOUT_RANGE_S: (f64, f64) = (0.0, 0.4);
const LOCALIZATION_BIAS_RANGE_M: (f64, f64) = (-1.0, 1.0);
const BRAKE_DELAY_RANGE_S: (f64, f64) = (0.0, 0.4);

/// The first vertical slice deliberately fixes the actor class.
/// Cyclists belong in a separate versioned family, not an ambiguous float axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossingActor {
    Pedestrian,
}

/// These values describe the provenance of the *range*, not a probability law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioRangeSemantics {
    SyntheticAssumption,
}

/// One complete candidate. Coordinates, units, clock and actor script are
/// explicit so an execution adapter need not infer their meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntersectionScenarioCandidate {
    pub schema_version: String,
    /// BLAKE3 identity over schema, policy, seed/budget, exact factor bits and fixed geometry.
    pub candidate_id: String,
    pub seed: u64,
    pub ordinal: u32,
    pub generation_budget: u32,
    pub generator_policy_id: String,
    pub road_geometry_id: String,
    /// Ego origin: x=-40 m, y=0 m; +x is direction of travel.
    pub ego_spawn_x_m: f64,
    pub ego_lane_center_y_m: f64,
    pub ego_initial_speed_mps: f64,
    /// Actor origin: x=0 m, y=-6 m; actor walks along +y across the lane.
    pub crossing_actor: CrossingActor,
    pub actor_spawn_x_m: f64,
    pub actor_spawn_y_m: f64,
    pub actor_crossing_speed_mps: f64,
    pub actor_crossing_path_length_m: f64,
    pub actor_crossing_start_s: f64,
    pub actor_crossing_direction_y: i8,
    /// Fraction of the actor's synthetic observation mask occluded (0..0.9).
    pub occlusion_fraction: f64,
    pub sensor_delay_s: f64,
    /// Deterministic dropout duration beginning at actor_crossing_start_s.
    pub sensor_dropout_duration_s: f64,
    /// Lateral localization estimate bias, not physical vehicle displacement.
    pub localization_lateral_bias_m: f64,
    /// Delay between a brake command and the simulated brake response.
    pub brake_actuation_delay_s: f64,
    pub timestep_s: f64,
    pub horizon_s: f64,
    pub range_semantics: ScenarioRangeSemantics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionGenerationError {
    ZeroBudget,
    BudgetTooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterexampleMinimizationError {
    InvalidSourceCandidate,
    EmptyPredicateId,
    ZeroEvaluationBudget,
    EvaluationBudgetTooLarge,
    OriginalNotCounterexample,
    InvalidEvaluationReceipt,
    InconsistentEvaluatorIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CounterexampleMinimizationTermination {
    CoordinatePassCompleted,
    EvaluationBudgetExhausted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleEvaluationReceipt {
    pub candidate_id: String,
    pub candidate: IntersectionScenarioCandidate,
    pub predicate_id: String,
    pub evaluator_id: String,
    pub execution_receipt_id: String,
    pub execution_digest: String,
    pub predicate_holds: bool,
}

impl CounterexampleEvaluationReceipt {
    pub fn is_valid_for(
        &self,
        candidate: &IntersectionScenarioCandidate,
        predicate_id: &str,
    ) -> bool {
        self.candidate_id == candidate.candidate_id
            && &self.candidate == candidate
            && self.candidate.identity_is_valid()
            && self.predicate_id == predicate_id
            && !self.evaluator_id.trim().is_empty()
            && !self.execution_receipt_id.trim().is_empty()
            && self.execution_digest.len() == 64
            && self.execution_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionFactorChange {
    pub factor_name: String,
    pub original_value: f64,
    pub minimized_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionCounterexampleMinimization {
    pub schema_version: String,
    pub parent_candidate_id: String,
    pub parent_candidate: IntersectionScenarioCandidate,
    pub predicate_id: String,
    pub minimization_policy_id: String,
    pub evaluation_receipts: Vec<CounterexampleEvaluationReceipt>,
    pub evaluation_budget: u32,
    pub evaluations: u32,
    pub failure_preserved: bool,
    pub termination: CounterexampleMinimizationTermination,
    pub factor_changes: Vec<IntersectionFactorChange>,
    pub minimized_candidate: IntersectionScenarioCandidate,
    /// BLAKE3 over the report with this field empty; not a signature.
    pub report_digest: String,
}

impl IntersectionCounterexampleMinimization {
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != INTERSECTION_CANDIDATE_SCHEMA
            || self.parent_candidate_id.trim().is_empty()
            || self.predicate_id.trim().is_empty()
            || self.minimization_policy_id != COUNTEREXAMPLE_MINIMIZATION_POLICY_ID
            || self.parent_candidate.candidate_id != self.parent_candidate_id
            || !self.parent_candidate.identity_is_valid()
            || self.evaluation_receipts.len() != self.evaluations as usize
            || self.evaluation_receipts.is_empty()
            || self.evaluation_receipts[0].candidate_id != self.parent_candidate_id
            || self.evaluation_receipts[0].candidate != self.parent_candidate
            || !self.evaluation_receipts[0].predicate_holds
            || !self.evaluation_receipts.iter().all(|receipt| {
                !receipt.evaluator_id.trim().is_empty()
                    && receipt.evaluator_id == self.evaluation_receipts[0].evaluator_id
                    && !receipt.execution_receipt_id.trim().is_empty()
                    && receipt.is_valid_for(&receipt.candidate, &self.predicate_id)
            })
            || !self.evaluation_receipts.iter().any(|receipt| {
                receipt.candidate_id == self.minimized_candidate.candidate_id
                    && receipt.candidate == self.minimized_candidate
                    && receipt.predicate_holds
            })
            || self.termination != (if self.evaluations >= self.evaluation_budget {
                CounterexampleMinimizationTermination::EvaluationBudgetExhausted
            } else {
                CounterexampleMinimizationTermination::CoordinatePassCompleted
            })
            || !factor_changes_are_consistent(
                &self.parent_candidate,
                &self.minimized_candidate,
                &self.factor_changes,
            )
            || self.evaluation_budget == 0
            || self.evaluation_budget > MAX_COUNTEREXAMPLE_MINIMIZATION_EVALUATIONS
            || self.evaluations == 0
            || self.evaluations > self.evaluation_budget
            || !self.failure_preserved
            || !self.minimized_candidate.identity_is_valid()
            || self.report_digest.len() != 64
            || !self.report_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return false;
        }

        let mut report = self.clone();
        let expected = report.report_digest.clone();
        report.report_digest.clear();
        serde_json::to_vec(&report)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected)
            .unwrap_or(false)
    }
}

/// Generate a deterministic Latin-hypercube-style candidate set.
///
/// Each factor is independently stratified into `budget` bins, then assigned
/// a seed-specific affine permutation and deterministic within-bin jitter.
/// This is a search-design measure only: it does not imply that candidate
/// frequency estimates real-world scenario prevalence.
pub fn generate_stratified_intersection_candidates(
    seed: u64,
    budget: u32,
) -> Result<Vec<IntersectionScenarioCandidate>, IntersectionGenerationError> {
    validate_budget(budget)?;
    let mut candidates = Vec::with_capacity(budget as usize);
    for ordinal in 0..budget {
        let factors = [
            stratified_value(seed, ordinal, budget, 0, EGO_SPEED_RANGE_MPS),
            stratified_value(seed, ordinal, budget, 1, CROSSING_START_RANGE_S),
            stratified_value(seed, ordinal, budget, 2, OCCLUSION_RANGE),
            stratified_value(seed, ordinal, budget, 3, SENSOR_DELAY_RANGE_S),
            stratified_value(seed, ordinal, budget, 4, SENSOR_DROPOUT_RANGE_S),
            stratified_value(seed, ordinal, budget, 5, LOCALIZATION_BIAS_RANGE_M),
            stratified_value(seed, ordinal, budget, 6, BRAKE_DELAY_RANGE_S),
        ];
        candidates.push(candidate_from_factors(
            seed,
            ordinal,
            budget,
            STRATIFIED_GENERATOR_POLICY_ID,
            factors,
        ));
    }
    Ok(candidates)
}

/// Generate a deterministic pseudorandom candidate set using SplitMix64.
///
/// This is the simple-search baseline for equal-budget comparisons against
/// stratified generation. It is reproducible for the named v0.1 algorithm;
/// it does not represent a real-world probability distribution.
pub fn generate_seeded_random_intersection_candidates(
    seed: u64,
    budget: u32,
) -> Result<Vec<IntersectionScenarioCandidate>, IntersectionGenerationError> {
    validate_budget(budget)?;
    let mut rng = SplitMix64 { state: seed };
    let mut candidates = Vec::with_capacity(budget as usize);
    for ordinal in 0..budget {
        let factors = [
            rng.range(EGO_SPEED_RANGE_MPS),
            rng.range(CROSSING_START_RANGE_S),
            rng.range(OCCLUSION_RANGE),
            rng.range(SENSOR_DELAY_RANGE_S),
            rng.range(SENSOR_DROPOUT_RANGE_S),
            rng.range(LOCALIZATION_BIAS_RANGE_M),
            rng.range(BRAKE_DELAY_RANGE_S),
        ];
        candidates.push(candidate_from_factors(
            seed,
            ordinal,
            budget,
            RANDOM_GENERATOR_POLICY_ID,
            factors,
        ));
    }
    Ok(candidates)
}

/// Minimize a scenario while preserving a caller-defined failure predicate.
///
/// The callback is the execution boundary: in the intended workflow it asks
/// Symtropy to execute/replay the candidate and returns a receipt binding the
/// exact candidate ID, predicate ID, evaluator identity and execution digest.
/// This function does not run physics or infer failure labels itself.
///
/// The deterministic coordinate shrinker moves each factor toward a declared
/// nominal value. Where the target removes the failure, a bounded 12-step
/// binary search finds a smaller value along that coordinate. This is a local
/// shrinker, not a proof of global minimality; non-monotone predicates can
/// contain disconnected failure regions.
pub fn minimize_intersection_counterexample<F>(
    source: &IntersectionScenarioCandidate,
    predicate_id: impl Into<String>,
    maximum_evaluations: u32,
    mut evaluator: F,
) -> Result<IntersectionCounterexampleMinimization, CounterexampleMinimizationError>
where
    F: FnMut(&IntersectionScenarioCandidate) -> CounterexampleEvaluationReceipt,
{
    let predicate_id = predicate_id.into();
    if !source.identity_is_valid() {
        return Err(CounterexampleMinimizationError::InvalidSourceCandidate);
    }
    if predicate_id.trim().is_empty() {
        return Err(CounterexampleMinimizationError::EmptyPredicateId);
    }
    if maximum_evaluations == 0 {
        return Err(CounterexampleMinimizationError::ZeroEvaluationBudget);
    }
    if maximum_evaluations > MAX_COUNTEREXAMPLE_MINIMIZATION_EVALUATIONS {
        return Err(CounterexampleMinimizationError::EvaluationBudgetTooLarge);
    }

    let mut evaluation_receipts = Vec::with_capacity(maximum_evaluations as usize);
    let mut evaluator_identity = None;
    let source_preserves = evaluate_and_record(
        source,
        &predicate_id,
        &mut evaluator,
        &mut evaluator_identity,
        &mut evaluation_receipts,
    )?;
    let mut evaluations = 1_u32;
    if !source_preserves {
        return Err(CounterexampleMinimizationError::OriginalNotCounterexample);
    }

    let parent_candidate_id = source.candidate_id.clone();
    let original_values = factor_values(source);
    let targets = [
        midpoint(EGO_SPEED_RANGE_MPS),
        midpoint(CROSSING_START_RANGE_S),
        OCCLUSION_RANGE.0,
        SENSOR_DELAY_RANGE_S.0,
        SENSOR_DROPOUT_RANGE_S.0,
        midpoint(LOCALIZATION_BIAS_RANGE_M),
        BRAKE_DELAY_RANGE_S.0,
    ];
    let names = [
        "ego_initial_speed_mps",
        "actor_crossing_start_s",
        "occlusion_fraction",
        "sensor_delay_s",
        "sensor_dropout_duration_s",
        "localization_lateral_bias_m",
        "brake_actuation_delay_s",
    ];

    // Preserve the source generator identity. The report separately binds
    // the minimization algorithm; candidate identity continues to mean
    // original scenario provenance plus exact current factor values.
    let mut minimized = source.clone();

    for dimension in 0..targets.len() {
        if evaluations >= maximum_evaluations {
            break;
        }
        let current_value = factor_value(&minimized, dimension);
        let target_value = targets[dimension];
        if current_value.to_bits() == target_value.to_bits() {
            continue;
        }

        let mut target_candidate = minimized.clone();
        set_factor_value(&mut target_candidate, dimension, target_value);
        target_candidate.candidate_id = candidate_identity(&target_candidate);
        let target_preserves = evaluate_and_record(
            &target_candidate,
            &predicate_id,
            &mut evaluator,
            &mut evaluator_identity,
            &mut evaluation_receipts,
        )?;
        evaluations += 1;

        if target_preserves {
            minimized = target_candidate;
            continue;
        }

        // Current value is known to preserve the predicate; target value does not.
        let mut accepted = minimized;
        let mut rejected = target_candidate;
        for _ in 0..MAX_COORDINATE_SHRINK_STEPS {
            if evaluations >= maximum_evaluations {
                break;
            }
            let accepted_value = factor_value(&accepted, dimension);
            let rejected_value = factor_value(&rejected, dimension);
            let midpoint_value = (accepted_value + rejected_value) * 0.5;
            let mut proposal = accepted.clone();
            set_factor_value(&mut proposal, dimension, midpoint_value);
            proposal.candidate_id = candidate_identity(&proposal);
            let preserves = evaluate_and_record(
                &proposal,
                &predicate_id,
                &mut evaluator,
                &mut evaluator_identity,
                &mut evaluation_receipts,
            )?;
            evaluations += 1;

            if preserves {
                accepted = proposal;
            } else {
                rejected = proposal;
            }
        }
        minimized = accepted;
    }

    let final_values = factor_values(&minimized);
    let factor_changes = names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            (original_values[index].to_bits() != final_values[index].to_bits()).then(|| {
                IntersectionFactorChange {
                    factor_name: (*name).into(),
                    original_value: original_values[index],
                    minimized_value: final_values[index],
                }
            })
        })
        .collect::<Vec<_>>();

    let termination = if evaluations >= maximum_evaluations {
        CounterexampleMinimizationTermination::EvaluationBudgetExhausted
    } else {
        CounterexampleMinimizationTermination::CoordinatePassCompleted
    };

    let mut report = IntersectionCounterexampleMinimization {
        schema_version: INTERSECTION_CANDIDATE_SCHEMA.into(),
        parent_candidate_id,
        parent_candidate: source.clone(),
        predicate_id,
        minimization_policy_id: COUNTEREXAMPLE_MINIMIZATION_POLICY_ID.into(),
        evaluation_budget: maximum_evaluations,
        evaluations,
        failure_preserved: true,
        termination,
        factor_changes,
        evaluation_receipts,
        minimized_candidate: minimized,
        report_digest: String::new(),
    };
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|_| CounterexampleMinimizationError::InvalidSourceCandidate)?;
    report.report_digest = blake3::hash(&report_bytes).to_hex().to_string();
    Ok(report)
}

fn evaluate_and_record<F>(
    candidate: &IntersectionScenarioCandidate,
    predicate_id: &str,
    evaluator: &mut F,
    evaluator_identity: &mut Option<String>,
    receipts: &mut Vec<CounterexampleEvaluationReceipt>,
) -> Result<bool, CounterexampleMinimizationError>
where
    F: FnMut(&IntersectionScenarioCandidate) -> CounterexampleEvaluationReceipt,
{
    let receipt = evaluator(candidate);
    if !receipt.is_valid_for(candidate, predicate_id) {
        return Err(CounterexampleMinimizationError::InvalidEvaluationReceipt);
    }
    if let Some(expected) = evaluator_identity.as_ref() {
        if expected != &receipt.evaluator_id {
            return Err(CounterexampleMinimizationError::InconsistentEvaluatorIdentity);
        }
    } else {
        *evaluator_identity = Some(receipt.evaluator_id.clone());
    }
    let predicate_holds = receipt.predicate_holds;
    receipts.push(receipt);
    Ok(predicate_holds)
}

fn factor_changes_are_consistent(
    parent: &IntersectionScenarioCandidate,
    minimized: &IntersectionScenarioCandidate,
    changes: &[IntersectionFactorChange],
) -> bool {
    let names = [
        "ego_initial_speed_mps",
        "actor_crossing_start_s",
        "occlusion_fraction",
        "sensor_delay_s",
        "sensor_dropout_duration_s",
        "localization_lateral_bias_m",
        "brake_actuation_delay_s",
    ];
    let original = factor_values(parent);
    let final_values = factor_values(minimized);
    let mut expected = Vec::new();
    for (index, name) in names.iter().enumerate() {
        if original[index].to_bits() != final_values[index].to_bits() {
            expected.push((*name, original[index], final_values[index]));
        }
    }
    expected.len() == changes.len()
        && expected.iter().zip(changes).all(|(expected, change)| {
            change.factor_name.as_str() == expected.0
                && change.original_value.to_bits() == expected.1.to_bits()
                && change.minimized_value.to_bits() == expected.2.to_bits()
        })
}

fn midpoint(bounds: (f64, f64)) -> f64 {
    (bounds.0 + bounds.1) * 0.5
}

fn factor_values(candidate: &IntersectionScenarioCandidate) -> [f64; 7] {
    [
        candidate.ego_initial_speed_mps,
        candidate.actor_crossing_start_s,
        candidate.occlusion_fraction,
        candidate.sensor_delay_s,
        candidate.sensor_dropout_duration_s,
        candidate.localization_lateral_bias_m,
        candidate.brake_actuation_delay_s,
    ]
}

fn factor_value(candidate: &IntersectionScenarioCandidate, dimension: usize) -> f64 {
    factor_values(candidate)[dimension]
}

fn set_factor_value(
    candidate: &mut IntersectionScenarioCandidate,
    dimension: usize,
    value: f64,
) {
    match dimension {
        0 => candidate.ego_initial_speed_mps = value,
        1 => candidate.actor_crossing_start_s = value,
        2 => candidate.occlusion_fraction = value,
        3 => candidate.sensor_delay_s = value,
        4 => candidate.sensor_dropout_duration_s = value,
        5 => candidate.localization_lateral_bias_m = value,
        6 => candidate.brake_actuation_delay_s = value,
        _ => unreachable!("factor dimension is bounded by the seven-factor schema"),
    }
}

fn validate_budget(budget: u32) -> Result<(), IntersectionGenerationError> {
    if budget == 0 {
        return Err(IntersectionGenerationError::ZeroBudget);
    }
    if budget > MAX_INTERSECTION_CANDIDATES {
        return Err(IntersectionGenerationError::BudgetTooLarge);
    }
    Ok(())
}

fn candidate_from_factors(
    seed: u64,
    ordinal: u32,
    budget: u32,
    generator_policy_id: &str,
    factors: [f64; 7],
) -> IntersectionScenarioCandidate {
    let [
        ego_initial_speed_mps,
        actor_crossing_start_s,
        occlusion_fraction,
        sensor_delay_s,
        sensor_dropout_duration_s,
        localization_lateral_bias_m,
        brake_actuation_delay_s,
    ] = factors;
    let mut candidate = IntersectionScenarioCandidate {
        schema_version: INTERSECTION_CANDIDATE_SCHEMA.into(),
        candidate_id: String::new(),
        seed,
        ordinal,
        generation_budget: budget,
        generator_policy_id: generator_policy_id.into(),
        road_geometry_id: ROAD_GEOMETRY_ID.into(),
        ego_spawn_x_m: EGO_SPAWN_X_M,
        ego_lane_center_y_m: EGO_LANE_CENTER_Y_M,
        ego_initial_speed_mps,
        crossing_actor: CrossingActor::Pedestrian,
        actor_spawn_x_m: ACTOR_SPAWN_X_M,
        actor_spawn_y_m: ACTOR_SPAWN_Y_M,
        actor_crossing_speed_mps: ACTOR_CROSSING_SPEED_MPS,
        actor_crossing_path_length_m: ACTOR_CROSSING_PATH_LENGTH_M,
        actor_crossing_start_s,
        actor_crossing_direction_y: ACTOR_CROSSING_DIRECTION_Y,
        occlusion_fraction,
        sensor_delay_s,
        sensor_dropout_duration_s,
        localization_lateral_bias_m,
        brake_actuation_delay_s,
        timestep_s: SIMULATION_TIMESTEP_S,
        horizon_s: SIMULATION_HORIZON_S,
        range_semantics: ScenarioRangeSemantics::SyntheticAssumption,
    };
    candidate.candidate_id = candidate_identity(&candidate);
    candidate
}

impl IntersectionScenarioCandidate {
    /// Verify immutable scenario fields and the candidate's content identity.
    pub fn identity_is_valid(&self) -> bool {
        self.schema_version == INTERSECTION_CANDIDATE_SCHEMA
            && self.generation_budget > 0
            && self.generation_budget <= MAX_INTERSECTION_CANDIDATES
            && self.ordinal < self.generation_budget
            && (self.generator_policy_id == STRATIFIED_GENERATOR_POLICY_ID
                || self.generator_policy_id == RANDOM_GENERATOR_POLICY_ID)
            && self.road_geometry_id == ROAD_GEOMETRY_ID
            && self.ego_spawn_x_m.to_bits() == EGO_SPAWN_X_M.to_bits()
            && self.ego_lane_center_y_m.to_bits() == EGO_LANE_CENTER_Y_M.to_bits()
            && self.crossing_actor == CrossingActor::Pedestrian
            && self.actor_spawn_x_m.to_bits() == ACTOR_SPAWN_X_M.to_bits()
            && self.actor_spawn_y_m.to_bits() == ACTOR_SPAWN_Y_M.to_bits()
            && self.actor_crossing_speed_mps.to_bits() == ACTOR_CROSSING_SPEED_MPS.to_bits()
            && self.actor_crossing_path_length_m.to_bits()
                == ACTOR_CROSSING_PATH_LENGTH_M.to_bits()
            && self.actor_crossing_direction_y == ACTOR_CROSSING_DIRECTION_Y
            && self.timestep_s.to_bits() == SIMULATION_TIMESTEP_S.to_bits()
            && self.horizon_s.to_bits() == SIMULATION_HORIZON_S.to_bits()
            && self.range_semantics == ScenarioRangeSemantics::SyntheticAssumption
            && in_range(self.ego_initial_speed_mps, EGO_SPEED_RANGE_MPS)
            && in_range(self.actor_crossing_start_s, CROSSING_START_RANGE_S)
            && in_range(self.occlusion_fraction, OCCLUSION_RANGE)
            && in_range(self.sensor_delay_s, SENSOR_DELAY_RANGE_S)
            && in_range(self.sensor_dropout_duration_s, SENSOR_DROPOUT_RANGE_S)
            && in_range(self.localization_lateral_bias_m, LOCALIZATION_BIAS_RANGE_M)
            && in_range(self.brake_actuation_delay_s, BRAKE_DELAY_RANGE_S)
            && self.candidate_id == candidate_identity(self)
    }
}

fn in_range(value: f64, bounds: (f64, f64)) -> bool {
    value.is_finite() && value >= bounds.0 && value <= bounds.1
}

fn stratified_value(
    seed: u64,
    ordinal: u32,
    budget: u32,
    dimension: u64,
    bounds: (f64, f64),
) -> f64 {
    let n = budget as u64;
    let mixed = mix64(seed ^ dimension.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut multiplier = (mixed % n).max(1);
    while gcd(multiplier, n) != 1 {
        multiplier += 1;
        if multiplier >= n {
            multiplier = 1;
        }
    }
    let offset = mix64(seed.rotate_left(23) ^ dimension.wrapping_mul(0xD1B5_4A32_D192_ED03)) % n;
    let stratum = ((ordinal as u64 * multiplier) + offset) % n;

    let jitter_bits = mix64(
        seed ^ (ordinal as u64).wrapping_mul(0xA24B_AED4_963E_E407)
            ^ dimension.wrapping_mul(0x9FB2_1C65_1E98_DF25),
    );
    let unit = (jitter_bits >> 11) as f64 / ((1_u64 << 53) as f64);
    let quantile = (stratum as f64 + unit) / n as f64;
    bounds.0 + quantile * (bounds.1 - bounds.0)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        (value >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    fn range(&mut self, bounds: (f64, f64)) -> f64 {
        bounds.0 + self.next_unit() * (bounds.1 - bounds.0)
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn candidate_identity(candidate: &IntersectionScenarioCandidate) -> String {
    let mut hasher = Hasher::new();
    hasher.update(candidate.schema_version.as_bytes());
    hasher.update(&candidate.seed.to_le_bytes());
    hasher.update(&candidate.ordinal.to_le_bytes());
    hasher.update(&candidate.generation_budget.to_le_bytes());
    hasher.update(candidate.generator_policy_id.as_bytes());
    hasher.update(candidate.road_geometry_id.as_bytes());
    for value in [
        candidate.ego_spawn_x_m,
        candidate.ego_lane_center_y_m,
        candidate.ego_initial_speed_mps,
        candidate.actor_spawn_x_m,
        candidate.actor_spawn_y_m,
        candidate.actor_crossing_speed_mps,
        candidate.actor_crossing_path_length_m,
        candidate.actor_crossing_start_s,
        candidate.occlusion_fraction,
        candidate.sensor_delay_s,
        candidate.sensor_dropout_duration_s,
        candidate.localization_lateral_bias_m,
        candidate.brake_actuation_delay_s,
        candidate.timestep_s,
        candidate.horizon_s,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&[match candidate.crossing_actor {
        CrossingActor::Pedestrian => 1,
    }]);
    hasher.update(&candidate.actor_crossing_direction_y.to_le_bytes());
    hasher.update(&[match candidate.range_semantics {
        ScenarioRangeSemantics::SyntheticAssumption => 1,
    }]);
    format!("sam-intx-{}", hasher.finalize().to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn mock_receipt(
        candidate: &IntersectionScenarioCandidate,
        predicate_id: &str,
        predicate_holds: bool,
    ) -> CounterexampleEvaluationReceipt {
        let mut hasher = Hasher::new();
        hasher.update(b"sam-synthetic-test-evaluator-v1");
        hasher.update(candidate.candidate_id.as_bytes());
        hasher.update(predicate_id.as_bytes());
        hasher.update(&[u8::from(predicate_holds)]);
        CounterexampleEvaluationReceipt {
            candidate_id: candidate.candidate_id.clone(),
            candidate: candidate.clone(),
            predicate_id: predicate_id.into(),
            evaluator_id: "unit-test-synthetic-evaluator-v1".into(),
            execution_receipt_id: format!("test-execution-{}", candidate.candidate_id),
            execution_digest: hasher.finalize().to_hex().to_string(),
            predicate_holds,
        }
    }

    #[test]
    fn same_seed_and_budget_reproduce_exact_candidate_ids() {
        let first = generate_stratified_intersection_candidates(42, 128).unwrap();
        let second = generate_stratified_intersection_candidates(42, 128).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(IntersectionScenarioCandidate::identity_is_valid));
    }

    #[test]
    fn seeded_random_generator_reproduces_exact_candidates() {
        let first = generate_seeded_random_intersection_candidates(42, 128).unwrap();
        let second = generate_seeded_random_intersection_candidates(42, 128).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(IntersectionScenarioCandidate::identity_is_valid));
    }

    #[test]
    fn random_and_stratified_policies_have_distinct_candidate_identity() {
        let random = generate_seeded_random_intersection_candidates(42, 64).unwrap();
        let stratified = generate_stratified_intersection_candidates(42, 64).unwrap();
        assert!(random.iter().all(|candidate| {
            candidate.generator_policy_id == RANDOM_GENERATOR_POLICY_ID
                && candidate.identity_is_valid()
        }));
        assert!(stratified.iter().all(|candidate| {
            candidate.generator_policy_id == STRATIFIED_GENERATOR_POLICY_ID
                && candidate.identity_is_valid()
        }));
        assert_ne!(
            random.iter().map(|candidate| &candidate.candidate_id).collect::<Vec<_>>(),
            stratified.iter().map(|candidate| &candidate.candidate_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn random_generator_respects_all_declared_factor_bounds() {
        let candidates = generate_seeded_random_intersection_candidates(7, 512).unwrap();
        for candidate in candidates {
            assert!(candidate.identity_is_valid());
            assert!(in_range(candidate.ego_initial_speed_mps, EGO_SPEED_RANGE_MPS));
            assert!(in_range(candidate.actor_crossing_start_s, CROSSING_START_RANGE_S));
            assert!(in_range(candidate.occlusion_fraction, OCCLUSION_RANGE));
            assert!(in_range(candidate.sensor_delay_s, SENSOR_DELAY_RANGE_S));
            assert!(in_range(candidate.sensor_dropout_duration_s, SENSOR_DROPOUT_RANGE_S));
            assert!(in_range(candidate.localization_lateral_bias_m, LOCALIZATION_BIAS_RANGE_M));
            assert!(in_range(candidate.brake_actuation_delay_s, BRAKE_DELAY_RANGE_S));
        }
    }

    #[test]
    fn independent_seed_changes_the_factor_values() {
        let first = generate_stratified_intersection_candidates(42, 64).unwrap();
        let second = generate_stratified_intersection_candidates(43, 64).unwrap();
        let factors = |candidates: &[IntersectionScenarioCandidate]| {
            candidates
                .iter()
                .map(|candidate| [
                    candidate.ego_initial_speed_mps,
                    candidate.actor_crossing_start_s,
                    candidate.occlusion_fraction,
                    candidate.sensor_delay_s,
                    candidate.sensor_dropout_duration_s,
                    candidate.localization_lateral_bias_m,
                    candidate.brake_actuation_delay_s,
                ])
                .collect::<Vec<_>>()
        };
        assert_ne!(factors(&first), factors(&second));
    }

    #[test]
    fn each_generated_factor_covers_every_stratum_once() {
        let budget = 64_u32;
        let candidates = generate_stratified_intersection_candidates(42, budget).unwrap();
        let factors: [fn(&IntersectionScenarioCandidate) -> f64; 7] = [
            |candidate| candidate.ego_initial_speed_mps,
            |candidate| candidate.actor_crossing_start_s,
            |candidate| candidate.occlusion_fraction,
            |candidate| candidate.sensor_delay_s,
            |candidate| candidate.sensor_dropout_duration_s,
            |candidate| candidate.localization_lateral_bias_m,
            |candidate| candidate.brake_actuation_delay_s,
        ];
        let bounds = [
            EGO_SPEED_RANGE_MPS,
            CROSSING_START_RANGE_S,
            OCCLUSION_RANGE,
            SENSOR_DELAY_RANGE_S,
            SENSOR_DROPOUT_RANGE_S,
            LOCALIZATION_BIAS_RANGE_M,
            BRAKE_DELAY_RANGE_S,
        ];
        for (factor, range) in factors.iter().zip(bounds) {
            let strata = candidates
                .iter()
                .map(|candidate| {
                    let quantile = (factor(candidate) - range.0) / (range.1 - range.0);
                    ((quantile * budget as f64).floor() as u32).min(budget - 1)
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(strata.len(), budget as usize);
        }
    }

    #[test]
    fn stratification_handles_unit_prime_and_non_power_of_two_budgets() {
        let factors: [fn(&IntersectionScenarioCandidate) -> f64; 7] = [
            |candidate| candidate.ego_initial_speed_mps,
            |candidate| candidate.actor_crossing_start_s,
            |candidate| candidate.occlusion_fraction,
            |candidate| candidate.sensor_delay_s,
            |candidate| candidate.sensor_dropout_duration_s,
            |candidate| candidate.localization_lateral_bias_m,
            |candidate| candidate.brake_actuation_delay_s,
        ];
        let bounds = [
            EGO_SPEED_RANGE_MPS,
            CROSSING_START_RANGE_S,
            OCCLUSION_RANGE,
            SENSOR_DELAY_RANGE_S,
            SENSOR_DROPOUT_RANGE_S,
            LOCALIZATION_BIAS_RANGE_M,
            BRAKE_DELAY_RANGE_S,
        ];
        for budget in [1_u32, 2, 3, 7, 31, 127] {
            let candidates = generate_stratified_intersection_candidates(0, budget).unwrap();
            for (factor, range) in factors.iter().zip(bounds) {
                let strata = candidates
                    .iter()
                    .map(|candidate| {
                        let quantile = (factor(candidate) - range.0) / (range.1 - range.0);
                        ((quantile * budget as f64).floor() as u32).min(budget - 1)
                    })
                    .collect::<BTreeSet<_>>();
                assert_eq!(strata.len(), budget as usize, "budget={budget}");
            }
        }
    }

    #[test]
    fn all_candidates_declare_synthetic_ranges_and_bounded_factors() {
        let candidates = generate_stratified_intersection_candidates(7, 256).unwrap();
        for candidate in candidates {
            assert_eq!(candidate.range_semantics, ScenarioRangeSemantics::SyntheticAssumption);
            assert!(candidate.identity_is_valid());
            assert!(in_range(candidate.ego_initial_speed_mps, EGO_SPEED_RANGE_MPS));
            assert!(in_range(candidate.actor_crossing_start_s, CROSSING_START_RANGE_S));
            assert!(in_range(candidate.occlusion_fraction, OCCLUSION_RANGE));
            assert!(in_range(candidate.sensor_delay_s, SENSOR_DELAY_RANGE_S));
            assert!(in_range(candidate.sensor_dropout_duration_s, SENSOR_DROPOUT_RANGE_S));
            assert!(in_range(candidate.localization_lateral_bias_m, LOCALIZATION_BIAS_RANGE_M));
            assert!(in_range(candidate.brake_actuation_delay_s, BRAKE_DELAY_RANGE_S));
        }
    }

    #[test]
    fn candidate_identity_changes_when_a_factor_is_mutated() {
        let mut candidate = generate_stratified_intersection_candidates(9, 1).unwrap().remove(0);
        candidate.sensor_delay_s = (candidate.sensor_delay_s + 0.01).min(SENSOR_DELAY_RANGE_S.1);
        assert!(!candidate.identity_is_valid());
    }

    #[test]
    fn counterexample_minimizer_shrinks_factors_and_preserves_parent_lineage() {
        let source = candidate_from_factors(
            23,
            0,
            1,
            RANDOM_GENERATOR_POLICY_ID,
            [20.0, 2.4, 0.8, 0.2, 0.2, 0.4, 0.3],
        );
        let report = minimize_intersection_counterexample(
            &source,
            "synthetic-crossing-conflict-predicate-v1",
            128,
            |candidate| {
                mock_receipt(
                    candidate,
                    "synthetic-crossing-conflict-predicate-v1",
                    candidate.ego_initial_speed_mps >= 16.0
                        && candidate.occlusion_fraction >= 0.6
                        && candidate.sensor_delay_s >= 0.1,
                )
            },
        )
        .unwrap();

        assert_eq!(report.parent_candidate_id, source.candidate_id);
        assert_eq!(
            report.minimized_candidate.generator_policy_id,
            source.generator_policy_id
        );
        assert_eq!(
            report.minimization_policy_id,
            COUNTEREXAMPLE_MINIMIZATION_POLICY_ID
        );
        assert!(report.minimized_candidate.identity_is_valid());
        assert!(report.failure_preserved);
        assert!(report.evaluations <= 128);
        assert_eq!(report.evaluation_receipts.len(), report.evaluations as usize);
        assert!(report.evaluation_receipts.iter().all(|receipt| {
            receipt.predicate_id == "synthetic-crossing-conflict-predicate-v1"
                && !receipt.execution_receipt_id.is_empty()
        }));
        assert!(report.minimized_candidate.ego_initial_speed_mps >= 16.0);
        assert!(report.minimized_candidate.ego_initial_speed_mps - 16.0 < 0.01);
        assert!(report.minimized_candidate.occlusion_fraction >= 0.6);
        assert!(report.minimized_candidate.occlusion_fraction - 0.6 < 0.01);
        assert!(report.minimized_candidate.sensor_delay_s >= 0.1);
        assert!(report.minimized_candidate.sensor_delay_s - 0.1 < 0.01);
        assert!(report.factor_changes.len() >= 3);
        assert!(report.identity_is_valid());
    }

    #[test]
    fn counterexample_minimizer_rejects_invalid_source_and_non_failure() {
        let source = generate_stratified_intersection_candidates(4, 1).unwrap().remove(0);
        assert_eq!(
            minimize_intersection_counterexample(&source, "predicate-v1", 64, |candidate| {
                mock_receipt(candidate, "predicate-v1", false)
            })
                .unwrap_err(),
            CounterexampleMinimizationError::OriginalNotCounterexample
        );
        let mut invalid = source;
        invalid.candidate_id.push_str("-tampered");
        assert_eq!(
            minimize_intersection_counterexample(&invalid, "predicate-v1", 64, |candidate| {
                mock_receipt(candidate, "predicate-v1", true)
            })
                .unwrap_err(),
            CounterexampleMinimizationError::InvalidSourceCandidate
        );
    }

    #[test]
    fn counterexample_minimizer_fails_closed_on_invalid_budgets_and_predicate_ids() {
        let source = generate_stratified_intersection_candidates(5, 1).unwrap().remove(0);
        assert_eq!(
            minimize_intersection_counterexample(&source, "predicate-v1", 0, |candidate| {
                mock_receipt(candidate, "predicate-v1", true)
            })
                .unwrap_err(),
            CounterexampleMinimizationError::ZeroEvaluationBudget
        );
        assert_eq!(
            minimize_intersection_counterexample(
                &source,
                "predicate-v1",
                MAX_COUNTEREXAMPLE_MINIMIZATION_EVALUATIONS + 1,
                |candidate| mock_receipt(candidate, "predicate-v1", true)
            )
            .unwrap_err(),
            CounterexampleMinimizationError::EvaluationBudgetTooLarge
        );
        assert_eq!(
            minimize_intersection_counterexample(&source, " ", 64, |candidate| {
                mock_receipt(candidate, " ", true)
            }).unwrap_err(),
            CounterexampleMinimizationError::EmptyPredicateId
        );
    }

    #[test]
    fn counterexample_minimizer_rejects_receipt_with_matching_id_but_different_candidate() {
        let source = generate_stratified_intersection_candidates(13, 1).unwrap().remove(0);
        let error = minimize_intersection_counterexample(
            &source,
            "crossing-failure-v1",
            32,
            |candidate| {
                let mut receipt = mock_receipt(candidate, "crossing-failure-v1", true);
                receipt.candidate.sensor_delay_s =
                    (receipt.candidate.sensor_delay_s + 0.01).min(SENSOR_DELAY_RANGE_S.1);
                // Keep the original candidate ID to model a malformed receipt.
                receipt
            },
        )
        .unwrap_err();
        assert_eq!(error, CounterexampleMinimizationError::InvalidEvaluationReceipt);
    }

    #[test]
    fn counterexample_minimizer_rejects_unbound_execution_receipts() {
        let source = generate_stratified_intersection_candidates(11, 1).unwrap().remove(0);
        let error = minimize_intersection_counterexample(
            &source,
            "crossing-failure-v1",
            32,
            |candidate| {
                let mut receipt = mock_receipt(candidate, "crossing-failure-v1", true);
                receipt.candidate_id.push_str("-unbound");
                receipt
            },
        )
        .unwrap_err();
        assert_eq!(error, CounterexampleMinimizationError::InvalidEvaluationReceipt);
    }

    #[test]
    fn counterexample_minimizer_rejects_mixed_evaluator_builds() {
        let source = candidate_from_factors(
            5,
            0,
            1,
            RANDOM_GENERATOR_POLICY_ID,
            [20.0, 2.4, 0.8, 0.2, 0.2, 0.4, 0.3],
        );
        let mut calls = 0usize;
        let error = minimize_intersection_counterexample(
            &source,
            "crossing-failure-v1",
            32,
            |candidate| {
                calls += 1;
                let mut receipt = mock_receipt(candidate, "crossing-failure-v1", true);
                if calls > 1 {
                    receipt.evaluator_id = "different-simulator-build".into();
                }
                receipt
            },
        )
        .unwrap_err();
        assert_eq!(error, CounterexampleMinimizationError::InconsistentEvaluatorIdentity);
    }

    #[test]
    fn counterexample_minimizer_report_detects_recomputed_digest_tampering() {
        let source = candidate_from_factors(
            9,
            0,
            1,
            RANDOM_GENERATOR_POLICY_ID,
            [20.0, 2.4, 0.8, 0.2, 0.2, 0.4, 0.3],
        );
        let mut report = minimize_intersection_counterexample(
            &source,
            "crossing-conflict-v1",
            64,
            |candidate| mock_receipt(
                candidate,
                "crossing-conflict-v1",
                candidate.ego_initial_speed_mps >= 16.0,
            ),
        )
        .unwrap();
        report.evaluations += 1;
        report.report_digest.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!report.identity_is_valid());
    }

    #[test]
    fn invalid_budgets_fail_closed() {
        assert_eq!(
            generate_stratified_intersection_candidates(1, 0),
            Err(IntersectionGenerationError::ZeroBudget)
        );
        assert_eq!(
            generate_stratified_intersection_candidates(1, MAX_INTERSECTION_CANDIDATES + 1),
            Err(IntersectionGenerationError::BudgetTooLarge)
        );
        assert_eq!(
            generate_seeded_random_intersection_candidates(1, 0),
            Err(IntersectionGenerationError::ZeroBudget)
        );
        assert_eq!(
            generate_seeded_random_intersection_candidates(1, MAX_INTERSECTION_CANDIDATES + 1),
            Err(IntersectionGenerationError::BudgetTooLarge)
        );
    }

    #[test]
    fn candidate_ids_are_unique_within_a_budget() {
        let candidates = generate_stratified_intersection_candidates(123, 512).unwrap();
        let identities = candidates
            .iter()
            .map(|candidate| candidate.candidate_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(identities.len(), candidates.len());
    }
}
