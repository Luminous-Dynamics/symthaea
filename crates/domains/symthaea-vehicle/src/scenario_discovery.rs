// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic candidate generation for a synthetic road-crossing scenario.
//!
//! This module owns scenario *design*, not execution or vehicle physics. It
//! creates a reproducible stratified candidate set for Symtropy to execute.
//! Every range is a synthetic design assumption, not an empirical distribution
//! or a claim about real-world frequency.

use blake3::Hasher;
use serde::{Deserialize, Serialize};

pub const INTERSECTION_CANDIDATE_SCHEMA: &str =
    "symthaea.vehicle.intersection-candidate.v0.1";
const GENERATOR_POLICY_ID: &str = "stratified-affine-lhs-v0.1";
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
    /// BLAKE3 identity over the schema, exact factor bits and fixed geometry.
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
    if budget == 0 {
        return Err(IntersectionGenerationError::ZeroBudget);
    }
    if budget > MAX_INTERSECTION_CANDIDATES {
        return Err(IntersectionGenerationError::BudgetTooLarge);
    }

    let mut candidates = Vec::with_capacity(budget as usize);
    for ordinal in 0..budget {
        let ego_initial_speed_mps = stratified_value(
            seed,
            ordinal,
            budget,
            0,
            EGO_SPEED_RANGE_MPS,
        );
        let actor_crossing_start_s = stratified_value(
            seed,
            ordinal,
            budget,
            1,
            CROSSING_START_RANGE_S,
        );
        let occlusion_fraction =
            stratified_value(seed, ordinal, budget, 2, OCCLUSION_RANGE);
        let sensor_delay_s =
            stratified_value(seed, ordinal, budget, 3, SENSOR_DELAY_RANGE_S);
        let sensor_dropout_duration_s = stratified_value(
            seed,
            ordinal,
            budget,
            4,
            SENSOR_DROPOUT_RANGE_S,
        );
        let localization_lateral_bias_m = stratified_value(
            seed,
            ordinal,
            budget,
            5,
            LOCALIZATION_BIAS_RANGE_M,
        );
        let brake_actuation_delay_s =
            stratified_value(seed, ordinal, budget, 6, BRAKE_DELAY_RANGE_S);

        let mut candidate = IntersectionScenarioCandidate {
            schema_version: INTERSECTION_CANDIDATE_SCHEMA.into(),
            candidate_id: String::new(),
            seed,
            ordinal,
            generation_budget: budget,
            generator_policy_id: GENERATOR_POLICY_ID.into(),
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
        candidates.push(candidate);
    }
    Ok(candidates)
}

impl IntersectionScenarioCandidate {
    /// Verify immutable scenario fields and the candidate's content identity.
    pub fn identity_is_valid(&self) -> bool {
        self.schema_version == INTERSECTION_CANDIDATE_SCHEMA
            && self.generation_budget > 0
            && self.generation_budget <= MAX_INTERSECTION_CANDIDATES
            && self.ordinal < self.generation_budget
            && self.generator_policy_id == GENERATOR_POLICY_ID
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

    #[test]
    fn same_seed_and_budget_reproduce_exact_candidate_ids() {
        let first = generate_stratified_intersection_candidates(42, 128).unwrap();
        let second = generate_stratified_intersection_candidates(42, 128).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(IntersectionScenarioCandidate::identity_is_valid));
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
    fn invalid_budgets_fail_closed() {
        assert_eq!(
            generate_stratified_intersection_candidates(1, 0),
            Err(IntersectionGenerationError::ZeroBudget)
        );
        assert_eq!(
            generate_stratified_intersection_candidates(1, MAX_INTERSECTION_CANDIDATES + 1),
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
