// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Simultaneous finite-sample bounds for Monte Carlo candidate scores.
//!
//! Each action × scenario cell must contain the same frozen number of
//! independent bounded replications. We form one-sided Hoeffding bounds for
//! five metrics and allocate an overall miscoverage budget across every
//! action × scenario × metric using a union bound.
//!
//! The guarantee is conditional on the declared ranges and independence
//! across replication indices within each cell. Dependence across different
//! action cells (for example, common random numbers) is allowed by the union
//! bound, but this module does not compute paired action-difference bounds.
//! It does not certify that the scenario bank represents reality.

use std::collections::BTreeMap;
use std::fmt;

use super::scenario::{
    ScenarioDefinition, ScenarioOutcome, ScenarioSet,
};
use super::sequential::{
    OperationalAction, OperationalActionSet,
};

/// Closed finite range declared before any score replications are inspected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreRange {
    pub lower: f64,
    pub upper: f64,
}

impl ScoreRange {
    pub fn new(lower: f64, upper: f64) -> Result<Self, ScoreBoundsError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(ScoreBoundsError::InvalidScoreRange);
        }
        let width = upper - lower;
        if !width.is_finite() {
            return Err(ScoreBoundsError::InvalidScoreRange);
        }
        Ok(Self { lower, upper })
    }

    pub fn width(self) -> f64 {
        self.upper - self.lower
    }

    fn contains(self, value: f64) -> bool {
        value.is_finite() && value >= self.lower && value <= self.upper
    }

    fn clip(self, value: f64) -> f64 {
        value.max(self.lower).min(self.upper)
    }
}

/// Frozen finite ranges for the five score components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreMetricRanges {
    pub target_information: ScoreRange,
    pub model_discrimination_information: ScoreRange,
    pub representativeness: ScoreRange,
    pub target_utility: ScoreRange,
    pub participant_burden_seconds: ScoreRange,
}

impl ScoreMetricRanges {
    fn validate(self) -> Result<(), ScoreBoundsError> {
        for range in [
            self.target_information,
            self.model_discrimination_information,
            self.representativeness,
            self.target_utility,
            self.participant_burden_seconds,
        ] {
            if !range.lower.is_finite()
                || !range.upper.is_finite()
                || range.lower > range.upper
                || !range.width().is_finite()
            {
                return Err(ScoreBoundsError::InvalidScoreRange);
            }
        }
        if self.target_information.lower < 0.0
            || self.model_discrimination_information.lower < 0.0
            || self.representativeness.lower < 0.0
            || self.representativeness.upper > 1.0
            || self.participant_burden_seconds.lower < 0.0
        {
            return Err(ScoreBoundsError::InvalidScoreRange);
        }
        Ok(())
    }
}

/// Preregistered score-bound settings and caller provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreBoundsConfig {
    /// Familywise one-sided miscoverage budget across all action × scenario
    /// cells and all five score metrics. Must be in (0, 1).
    pub overall_delta: f64,
    /// Exact common replication count for every action × scenario cell.
    pub replications_per_cell: usize,
    /// Caller-provided digest of the simulation generator, score definition,
    /// ranges, common-random-number construction and independence contract.
    pub score_contract_digest: String,
    pub ranges: ScoreMetricRanges,
}

/// One Monte Carlo replication for one operational action in one scenario.
///
/// Replication indices must be exactly 0..replications_per_cell for every
/// action × scenario pair. Indices may be aligned across actions for common
/// random numbers; within each cell they must represent independent draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreReplication {
    pub action_id: u32,
    pub scenario_id: u16,
    pub replication_index: u32,
    pub operational_mapping_qualified: bool,
    pub model_adequacy_qualified: bool,
    pub target_information: f64,
    pub model_discrimination_information: f64,
    pub representativeness: f64,
    pub target_utility: f64,
    pub participant_burden_seconds: f64,
}

/// Replay record with all raw data and conservative scenario outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreBoundBatch {
    pub method: &'static str,
    pub action_set_digest: String,
    pub action_snapshot: Vec<OperationalAction>,
    pub scenario_set_digest: String,
    pub scenarios: Vec<ScenarioDefinition>,
    pub config: ScoreBoundsConfig,
    /// 5 × number_of_actions × number_of_scenarios.
    pub simultaneous_bound_count: usize,
    /// log(simultaneous_bound_count / overall_delta), computed in log space.
    pub log_m_over_delta: f64,
    pub raw_replications: Vec<ScoreReplication>,
    /// Directly consumable by aggregate_scenario_assessments.
    pub scenario_outcomes: Vec<ScenarioOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreBoundsError {
    InvalidScoreRange,
    InvalidOverallDelta,
    TooFewReplications,
    ReplicationCountOverflow,
    BoundCountOverflow,
    IncompleteReplicationGrid,
    UnknownActionId,
    UnknownScenarioId,
    DuplicateReplication,
    ReplicationIndexOutOfRange,
    ScoreOutsideDeclaredRange,
    MissingScoreContractDigest,
    NumericFailure,
}

impl fmt::Display for ScoreBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidScoreRange => "score ranges must be finite and have valid domains",
            Self::InvalidOverallDelta => "overall delta must be finite and in (0, 1)",
            Self::TooFewReplications => "at least two replications are required per action × scenario",
            Self::ReplicationCountOverflow => "action × scenario × replication count overflowed usize",
            Self::BoundCountOverflow => "simultaneous bound count overflowed usize",
            Self::IncompleteReplicationGrid => "every action × scenario cell must contain the exact frozen replication count",
            Self::UnknownActionId => "replication references an unknown action",
            Self::UnknownScenarioId => "replication references an unknown scenario",
            Self::DuplicateReplication => "duplicate action × scenario × replication index",
            Self::ReplicationIndexOutOfRange => "replication index is outside the frozen range",
            Self::ScoreOutsideDeclaredRange => "score value is outside its preregistered finite range",
            Self::MissingScoreContractDigest => "score-contract digest must not be empty",
            Self::NumericFailure => "score aggregation or confidence-bound arithmetic failed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ScoreBoundsError {}

/// Build simultaneous one-sided Hoeffding bounds for every action × scenario.
///
/// For each of the four higher-is-better metrics, the result contains a lower
/// bound. For participant burden, it contains an upper bound. If m is
/// 5 × |actions| × |scenarios| and delta is overall_delta, the radius is
///
///   (range width) × sqrt(log(m / delta) / (2 n)),
///
/// capped at the full declared range. Each one-sided tail receives delta/m,
/// so the union bound controls the probability that any reported bound fails
/// at no more than delta under the per-cell bounded-IID assumption.
pub fn bound_scenario_scores(
    action_set: &OperationalActionSet,
    scenario_set: &ScenarioSet,
    replications: &[ScoreReplication],
    config: &ScoreBoundsConfig,
) -> Result<ScoreBoundBatch, ScoreBoundsError> {
    config.ranges.validate()?;
    if !config.overall_delta.is_finite()
        || config.overall_delta <= 0.0
        || config.overall_delta >= 1.0
    {
        return Err(ScoreBoundsError::InvalidOverallDelta);
    }
    if config.replications_per_cell < 2 {
        return Err(ScoreBoundsError::TooFewReplications);
    }
    if config.replications_per_cell - 1 > u32::MAX as usize {
        return Err(ScoreBoundsError::ReplicationIndexOutOfRange);
    }
    if config.score_contract_digest.trim().is_empty() {
        return Err(ScoreBoundsError::MissingScoreContractDigest);
    }

    let cell_count = action_set
        .actions()
        .len()
        .checked_mul(scenario_set.scenarios().len())
        .ok_or(ScoreBoundsError::ReplicationCountOverflow)?;
    let expected_count = cell_count
        .checked_mul(config.replications_per_cell)
        .ok_or(ScoreBoundsError::ReplicationCountOverflow)?;
    if replications.len() != expected_count {
        return Err(ScoreBoundsError::IncompleteReplicationGrid);
    }
    let simultaneous_bound_count = cell_count
        .checked_mul(5)
        .ok_or(ScoreBoundsError::BoundCountOverflow)?;
    let log_m_over_delta =
        (simultaneous_bound_count as f64).ln() - config.overall_delta.ln();
    if !log_m_over_delta.is_finite() || log_m_over_delta <= 0.0 {
        return Err(ScoreBoundsError::NumericFailure);
    }
    let radius_factor =
        (log_m_over_delta / (2.0 * config.replications_per_cell as f64)).sqrt();
    if !radius_factor.is_finite() || radius_factor < 0.0 {
        return Err(ScoreBoundsError::NumericFailure);
    }

    let mut by_replication = BTreeMap::new();
    for replication in replications {
        if !action_set
            .actions()
            .iter()
            .any(|action| action.action_id == replication.action_id)
        {
            return Err(ScoreBoundsError::UnknownActionId);
        }
        if !scenario_set
            .scenarios()
            .iter()
            .any(|scenario| scenario.scenario_id == replication.scenario_id)
        {
            return Err(ScoreBoundsError::UnknownScenarioId);
        }
        if replication.replication_index as usize >= config.replications_per_cell {
            return Err(ScoreBoundsError::ReplicationIndexOutOfRange);
        }
        validate_replication(replication, config.ranges)?;
        if by_replication
            .insert(
                (
                    replication.action_id,
                    replication.scenario_id,
                    replication.replication_index,
                ),
                replication,
            )
            .is_some()
        {
            return Err(ScoreBoundsError::DuplicateReplication);
        }
    }

    if by_replication.len() != expected_count {
        return Err(ScoreBoundsError::IncompleteReplicationGrid);
    }

    let mut raw_replications = replications.to_vec();
    raw_replications.sort_by_key(|replication| {
        (
            replication.action_id,
            replication.scenario_id,
            replication.replication_index,
        )
    });

    let mut scenario_outcomes = Vec::with_capacity(cell_count);
    for action in action_set.actions() {
        for scenario in scenario_set.scenarios() {
            let mut target_information = 0.0;
            let mut model_discrimination_information = 0.0;
            let mut representativeness = 0.0;
            let mut target_utility = 0.0;
            let mut participant_burden_seconds = 0.0;
            let mut mapping_qualified = true;
            let mut model_adequacy_qualified = true;

            for replication_index in 0..config.replications_per_cell {
                let replication = by_replication
                    .get(&(
                        action.action_id,
                        scenario.scenario_id,
                        replication_index as u32,
                    ))
                    .copied()
                    .ok_or(ScoreBoundsError::IncompleteReplicationGrid)?;
                let count = (replication_index + 1) as f64;
                target_information = update_mean(
                    target_information,
                    replication.target_information,
                    count,
                );
                model_discrimination_information = update_mean(
                    model_discrimination_information,
                    replication.model_discrimination_information,
                    count,
                );
                representativeness = update_mean(
                    representativeness,
                    replication.representativeness,
                    count,
                );
                target_utility = update_mean(
                    target_utility,
                    replication.target_utility,
                    count,
                );
                participant_burden_seconds = update_mean(
                    participant_burden_seconds,
                    replication.participant_burden_seconds,
                    count,
                );
                mapping_qualified &= replication.operational_mapping_qualified;
                model_adequacy_qualified &= replication.model_adequacy_qualified;
            }

            let target_information_lower_bound = one_sided_lower_bound(
                target_information,
                config.ranges.target_information,
                radius_factor,
            );
            let model_discrimination_information_lower_bound = one_sided_lower_bound(
                model_discrimination_information,
                config.ranges.model_discrimination_information,
                radius_factor,
            );
            let representativeness_lower_bound = one_sided_lower_bound(
                representativeness,
                config.ranges.representativeness,
                radius_factor,
            );
            let target_utility_lower_bound = one_sided_lower_bound(
                target_utility,
                config.ranges.target_utility,
                radius_factor,
            );
            let participant_burden_upper_bound = one_sided_upper_bound(
                participant_burden_seconds,
                config.ranges.participant_burden_seconds,
                radius_factor,
            );

            scenario_outcomes.push(ScenarioOutcome {
                action_id: action.action_id,
                scenario_id: scenario.scenario_id,
                operational_mapping_qualified: mapping_qualified,
                model_adequacy_qualified,
                target_information,
                target_information_lower_bound,
                model_discrimination_information,
                model_discrimination_information_lower_bound,
                representativeness,
                representativeness_lower_bound,
                target_utility,
                target_utility_lower_bound,
                participant_burden_seconds,
                participant_burden_upper_bound,
            });
        }
    }

    Ok(ScoreBoundBatch {
        method: "simultaneous-one-sided-hoeffding-v1",
        action_set_digest: action_set.digest().to_owned(),
        action_snapshot: action_set.actions().to_vec(),
        scenario_set_digest: scenario_set.digest().to_owned(),
        scenarios: scenario_set.scenarios().to_vec(),
        config: config.clone(),
        simultaneous_bound_count,
        log_m_over_delta,
        raw_replications,
        scenario_outcomes,
    })
}

fn validate_replication(
    replication: &ScoreReplication,
    ranges: ScoreMetricRanges,
) -> Result<(), ScoreBoundsError> {
    if !ranges.target_information.contains(replication.target_information)
        || !ranges
            .model_discrimination_information
            .contains(replication.model_discrimination_information)
        || !ranges
            .representativeness
            .contains(replication.representativeness)
        || !ranges.target_utility.contains(replication.target_utility)
        || !ranges
            .participant_burden_seconds
            .contains(replication.participant_burden_seconds)
    {
        return Err(ScoreBoundsError::ScoreOutsideDeclaredRange);
    }
    Ok(())
}

/// Stable online mean for values inside a finite declared interval.
fn update_mean(previous_mean: f64, value: f64, count: f64) -> f64 {
    previous_mean + (value - previous_mean) / count
}

fn one_sided_lower_bound(mean: f64, range: ScoreRange, radius_factor: f64) -> f64 {
    let radius = (range.width() * radius_factor).min(range.width());
    range.clip(mean - radius)
}

fn one_sided_upper_bound(mean: f64, range: ScoreRange, radius_factor: f64) -> f64 {
    let radius = (range.width() * radius_factor).min(range.width());
    range.clip(mean + radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::scenario::{
        aggregate_scenario_assessments, RobustUtility, ScenarioDefinition,
    };
    use super::super::sequential::{
        OperationalAction, PolicyHistory, SequentialPolicyConfig, StimulusClass,
    };
    use std::collections::BTreeMap;

    fn actions() -> OperationalActionSet {
        OperationalActionSet::new(
            vec![
                OperationalAction {
                    action_id: 2,
                    audit_stratum: 1,
                    condition_id: 0,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 1.0,
                    response_scale_id: 1,
                    rt_deadline_ms: Some(1_000),
                },
                OperationalAction {
                    action_id: 1,
                    audit_stratum: 1,
                    condition_id: 0,
                    stimulus_class: StimulusClass::Noise,
                    stimulus_level: 0.0,
                    response_scale_id: 1,
                    rt_deadline_ms: Some(1_000),
                },
            ],
            "sha256:action-set",
        )
        .unwrap()
    }

    fn scenarios() -> ScenarioSet {
        ScenarioSet::new(
            vec![
                ScenarioDefinition {
                    scenario_id: 20,
                    weight: 0.3,
                },
                ScenarioDefinition {
                    scenario_id: 10,
                    weight: 0.7,
                },
            ],
            "sha256:scenario-set",
        )
        .unwrap()
    }

    fn ranges() -> ScoreMetricRanges {
        ScoreMetricRanges {
            target_information: ScoreRange { lower: 0.0, upper: 10.0 },
            model_discrimination_information: ScoreRange { lower: 0.0, upper: 10.0 },
            representativeness: ScoreRange { lower: 0.0, upper: 1.0 },
            target_utility: ScoreRange { lower: -10.0, upper: 10.0 },
            participant_burden_seconds: ScoreRange { lower: 0.0, upper: 10.0 },
        }
    }

    fn config(n: usize) -> ScoreBoundsConfig {
        ScoreBoundsConfig {
            overall_delta: 0.05,
            replications_per_cell: n,
            score_contract_digest: "sha256:score-contract".to_owned(),
            ranges: ranges(),
        }
    }

    fn replications(n: usize) -> Vec<ScoreReplication> {
        let mut records = Vec::new();
        for action_id in [1, 2] {
            for scenario_id in [10, 20] {
                for replication_index in 0..n {
                    let perturbation = (replication_index % 7) as f64 * 0.01;
                    let favorable = action_id == 2 && scenario_id == 10;
                    records.push(ScoreReplication {
                        action_id,
                        scenario_id,
                        replication_index: replication_index as u32,
                        operational_mapping_qualified: true,
                        model_adequacy_qualified: true,
                        target_information: (if favorable { 7.0 } else { 5.0 }) + perturbation,
                        model_discrimination_information: 3.0 + perturbation,
                        representativeness: if scenario_id == 20 { 0.65 } else { 0.85 },
                        target_utility: (if favorable { 4.0 } else { 2.0 }) + perturbation,
                        participant_burden_seconds: 1.0 + perturbation,
                    });
                }
            }
        }
        records
    }

    #[test]
    fn computes_one_sided_hoeffding_reference_bounds() {
        let action_set = actions();
        let scenario_set = ScenarioSet::new(
            vec![ScenarioDefinition { scenario_id: 1, weight: 1.0 }],
            "sha256:one-scenario",
        )
        .unwrap();
        let n = 100usize;
        let mut records = Vec::new();
        for replication_index in 0..n {
            records.push(ScoreReplication {
                action_id: 1,
                scenario_id: 1,
                replication_index: replication_index as u32,
                operational_mapping_qualified: true,
                model_adequacy_qualified: true,
                target_information: 5.0,
                model_discrimination_information: 4.0,
                representativeness: 0.8,
                target_utility: 2.0,
                participant_burden_seconds: 3.0,
            });
        }
        let one_action = OperationalActionSet::new(
            vec![action_set.actions()[0].clone()],
            "sha256:one-action",
        )
        .unwrap();
        let mut cfg = config(n);
        cfg.overall_delta = 0.05;
        let batch = bound_scenario_scores(&one_action, &scenario_set, &records, &cfg).unwrap();
        let factor = (5.0_f64.ln() - 0.05_f64.ln() + 0.0).sqrt()
            / (2.0 * n as f64).sqrt();
        let expected_lower = (5.0 - 10.0 * factor).max(0.0);
        assert!((batch.scenario_outcomes[0].target_information_lower_bound - expected_lower).abs() < 1.0e-12);
        assert!((batch.scenario_outcomes[0].target_information - 5.0).abs() < 1.0e-12);
        assert!(batch.scenario_outcomes[0].participant_burden_upper_bound >= 3.0);
        assert_eq!(batch.simultaneous_bound_count, 5);
    }

    #[test]
    fn simultaneous_multiplicity_widens_bounds_as_action_scenario_grid_grows() {
        let small_actions = OperationalActionSet::new(
            vec![actions().actions()[0].clone()],
            "sha256:one-action",
        )
        .unwrap();
        let small_scenarios = ScenarioSet::new(
            vec![ScenarioDefinition { scenario_id: 10, weight: 1.0 }],
            "sha256:one-scenario",
        )
        .unwrap();
        let n = 100;
        let small_records = (0..n)
            .map(|replication_index| ScoreReplication {
                action_id: small_actions.actions()[0].action_id,
                scenario_id: 10,
                replication_index: replication_index as u32,
                operational_mapping_qualified: true,
                model_adequacy_qualified: true,
                target_information: 5.0,
                model_discrimination_information: 5.0,
                representativeness: 0.8,
                target_utility: 2.0,
                participant_burden_seconds: 3.0,
            })
            .collect::<Vec<_>>();
        let small = bound_scenario_scores(
            &small_actions,
            &small_scenarios,
            &small_records,
            &config(n),
        )
        .unwrap();
        let large = bound_scenario_scores(
            &actions(),
            &scenarios(),
            &replications(n),
            &config(n),
        )
        .unwrap();
        assert_eq!(small.simultaneous_bound_count, 5);
        assert_eq!(large.simultaneous_bound_count, 20);
        assert!(large.log_m_over_delta > small.log_m_over_delta);
        assert!(
            large.scenario_outcomes[0].target_information_lower_bound
                < small.scenario_outcomes[0].target_information_lower_bound
        );
    }

    #[test]
    fn canonical_results_are_independent_of_input_record_order() {
        let mut reversed = replications(10);
        let forward = bound_scenario_scores(&actions(), &scenarios(), &reversed, &config(10)).unwrap();
        reversed.reverse();
        let backward = bound_scenario_scores(&actions(), &scenarios(), &reversed, &config(10)).unwrap();
        assert_eq!(forward, backward);
    }

    #[test]
    fn missing_duplicate_unknown_and_out_of_range_records_fail_closed() {
        let action_set = actions();
        let scenario_set = scenarios();
        let valid = replications(4);

        let mut missing = valid.clone();
        missing.pop();
        assert_eq!(
            bound_scenario_scores(&action_set, &scenario_set, &missing, &config(4)),
            Err(ScoreBoundsError::IncompleteReplicationGrid)
        );

        let mut duplicate = valid.clone();
        duplicate[1] = duplicate[0];
        assert_eq!(
            bound_scenario_scores(&action_set, &scenario_set, &duplicate, &config(4)),
            Err(ScoreBoundsError::DuplicateReplication)
        );

        let mut unknown = valid.clone();
        unknown[0].action_id = 999;
        assert_eq!(
            bound_scenario_scores(&action_set, &scenario_set, &unknown, &config(4)),
            Err(ScoreBoundsError::UnknownActionId)
        );

        let mut bad_index = valid.clone();
        bad_index[0].replication_index = 99;
        assert_eq!(
            bound_scenario_scores(&action_set, &scenario_set, &bad_index, &config(4)),
            Err(ScoreBoundsError::ReplicationIndexOutOfRange)
        );
    }

    #[test]
    fn invalid_ranges_delta_and_score_contract_fail_closed() {
        assert_eq!(
            ScoreRange::new(f64::NAN, 1.0),
            Err(ScoreBoundsError::InvalidScoreRange)
        );
        let mut invalid_delta = config(2);
        invalid_delta.overall_delta = 1.0;
        assert_eq!(
            bound_scenario_scores(&actions(), &scenarios(), &replications(2), &invalid_delta),
            Err(ScoreBoundsError::InvalidOverallDelta)
        );
        let mut missing_contract = config(2);
        missing_contract.score_contract_digest = " ".to_owned();
        assert_eq!(
            bound_scenario_scores(&actions(), &scenarios(), &replications(2), &missing_contract),
            Err(ScoreBoundsError::MissingScoreContractDigest)
        );
        let mut out_of_range = replications(2);
        out_of_range[0].representativeness = 1.2;
        assert_eq!(
            bound_scenario_scores(&actions(), &scenarios(), &out_of_range, &config(2)),
            Err(ScoreBoundsError::ScoreOutsideDeclaredRange)
        );
    }

    #[test]
    fn all_replication_qualification_flags_must_pass() {
        let mut records = replications(5);
        records[0].operational_mapping_qualified = false;
        records[3].model_adequacy_qualified = false;
        let batch = bound_scenario_scores(&actions(), &scenarios(), &records, &config(5)).unwrap();
        let first = batch
            .scenario_outcomes
            .iter()
            .find(|outcome| outcome.action_id == records[0].action_id && outcome.scenario_id == records[0].scenario_id)
            .unwrap();
        assert!(!first.operational_mapping_qualified);
        assert!(!first.model_adequacy_qualified);
    }

    #[test]
    fn score_bounds_feed_the_existing_scenario_aggregator() {
        let action_set = actions();
        let scenario_set = scenarios();
        let bounded = bound_scenario_scores(
            &action_set,
            &scenario_set,
            &replications(100),
            &config(100),
        )
        .unwrap();
        let aggregated = aggregate_scenario_assessments(
            &action_set,
            &scenario_set,
            &bounded.scenario_outcomes,
            RobustUtility::LowerTailCvar { tail_mass: 0.5 },
            bounded.config.score_contract_digest.clone(),
        )
        .unwrap();
        assert_eq!(aggregated.assessments.len(), action_set.actions().len());
        assert_eq!(aggregated.score_contract_digest, "sha256:score-contract");

        let history = PolicyHistory::new(
            BTreeMap::new(),
            "sha256:history",
            &action_set,
        )
        .unwrap();
        let policy_config = SequentialPolicyConfig {
            policy_version: "spectra-test-v1".to_owned(),
            model_set_version: "scenario-bank-v1".to_owned(),
            assessment_set_digest: "sha256:assessment-batch".to_owned(),
            seed: 12,
            decision_index: 0,
            exploration_probability: 0.0,
            minimum_representativeness: 0.0,
            minimum_model_discrimination_information: Some(0.0),
            audit_floor_by_stratum: BTreeMap::new(),
            remaining_participant_time_seconds: Some(10.0),
        };
        let selected = super::super::sequential::select_next_action(
            &action_set,
            &aggregated.assessments,
            &history,
            &policy_config,
        )
        .unwrap();
        assert!(selected.receipt.selection_probability > 0.0);
    }

    #[test]
    fn refuses_a_single_replication_even_if_it_is_finite() {
        assert_eq!(
            bound_scenario_scores(&actions(), &scenarios(), &replications(1), &config(1)),
            Err(ScoreBoundsError::TooFewReplications)
        );
    }
}
