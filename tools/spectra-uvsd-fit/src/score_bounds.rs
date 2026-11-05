// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Simultaneous finite-sample Hoeffding bounds for scenario score estimates.
//!
//! Each score series must consist of independent Monte Carlo replications
//! bounded by a frozen range. A Bonferroni allocation controls familywise
//! error across every action × scenario × metric claim, without assuming
//! independence between different score claims. These bounds quantify Monte
//! Carlo uncertainty only; they do not correct simulator or model bias.

use std::collections::BTreeMap;
use std::fmt;

use super::scenario::{ScenarioDefinition, ScenarioOutcome, ScenarioSet};
use super::sequential::{OperationalAction, OperationalActionSet};

pub const SCORE_BOUND_METHOD_ID: &str = "hoeffding-bonferroni-one-sided-v1";
pub const ANYTIME_SCORE_BOUND_METHOD_ID: &str = "anytime-hoeffding-alpha-spending-v1";
pub const ANYTIME_ALPHA_SPENDING_RULE_ID: &str = "n-times-n-plus-one-summable-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundSchedule {
    FixedLook,
    Anytime,
}

/// A declared finite range for one Monte Carlo score family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreRange {
    pub lower: f64,
    pub upper: f64,
}

impl ScoreRange {
    fn validate(self) -> Result<(), ScoreBoundError> {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || self.lower > self.upper
            || !(self.upper - self.lower).is_finite()
        {
            return Err(ScoreBoundError::InvalidScoreRange);
        }
        Ok(())
    }

    fn width(self) -> f64 {
        self.upper - self.lower
    }
}

/// All five ranges used in a simultaneous candidate assessment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreRanges {
    pub target_information: ScoreRange,
    pub model_discrimination_information: ScoreRange,
    pub representativeness: ScoreRange,
    pub target_utility: ScoreRange,
    pub participant_burden_seconds: ScoreRange,
}

impl ScoreRanges {
    fn validate(self) -> Result<(), ScoreBoundError> {
        self.target_information.validate()?;
        self.model_discrimination_information.validate()?;
        self.representativeness.validate()?;
        self.target_utility.validate()?;
        self.participant_burden_seconds.validate()?;
        if self.target_information.lower < 0.0
            || self.model_discrimination_information.lower < 0.0
            || self.representativeness.lower < 0.0
            || self.representativeness.upper > 1.0
            || self.participant_burden_seconds.lower < 0.0
        {
            return Err(ScoreBoundError::InvalidScoreRange);
        }
        Ok(())
    }
}

/// Independent Monte Carlo score replicates for one action × scenario pair.
///
/// The caller must ensure each vector consists of independent bounded
/// replications from the frozen score-generation procedure. Different metric
/// vectors may be dependent; Bonferroni controls the joint claims regardless.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedScoreSamples {
    pub action_id: u32,
    pub scenario_id: u16,
    pub operational_mapping_qualified: bool,
    pub model_adequacy_qualified: bool,
    pub target_information: Vec<f64>,
    pub model_discrimination_information: Vec<f64>,
    pub representativeness: Vec<f64>,
    pub target_utility: Vec<f64>,
    pub participant_burden_seconds: Vec<f64>,
}

/// Configuration and provenance for a simultaneous score-bound campaign.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreBoundsConfig {
    /// Familywise one-sided error probability across all claims.
    pub familywise_alpha: f64,
    /// Minimum number of independent replications in every metric series.
    pub minimum_replicates: usize,
    pub ranges: ScoreRanges,
    /// Caller-supplied commitment to the score-generation/estimand contract.
    pub score_contract_digest: String,
    /// Caller-supplied source/generator revision or archive commitment.
    pub source_digest: String,
    pub root_seed: u64,
}

/// Point estimate and finite-sample one-sided or two-sided score bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricScoreBound {
    pub point_estimate: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub sample_count: usize,
    pub declared_range: ScoreRange,
}

/// Complete bounded-score campaign receipt. All raw replicate vectors are
/// retained and sorted by (action_id, scenario_id) for deterministic replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreBoundsReceipt {
    pub method_id: &'static str,
    /// Whether bounds are simultaneous over every replication count.
    pub time_uniform: bool,
    /// "none" for fixed-look intervals; a stable rule ID for anytime bounds.
    pub alpha_spending_rule: &'static str,
    /// Canonical action inputs addressed by action_set_digest.
    pub action_set_digest: String,
    pub action_snapshot: Vec<OperationalAction>,
    /// Canonical normalized scenario definitions addressed by scenario_set_digest.
    pub scenario_set_digest: String,
    pub scenarios: Vec<ScenarioDefinition>,
    pub familywise_alpha: f64,
    pub per_claim_alpha: f64,
    pub familywise_confidence: f64,
    pub claim_count: usize,
    pub minimum_replicates: usize,
    pub ranges: ScoreRanges,
    pub score_contract_digest: String,
    pub source_digest: String,
    pub root_seed: u64,
    pub samples: Vec<BoundedScoreSamples>,
    pub metrics: Vec<ActionScenarioMetricBounds>,
    pub scenario_outcomes: Vec<ScenarioOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionScenarioMetricBounds {
    pub action_id: u32,
    pub scenario_id: u16,
    pub target_information: MetricScoreBound,
    pub model_discrimination_information: MetricScoreBound,
    pub representativeness: MetricScoreBound,
    pub target_utility: MetricScoreBound,
    pub participant_burden_seconds: MetricScoreBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreBoundError {
    InvalidFamilywiseAlpha,
    InvalidMinimumReplicates,
    InvalidScoreRange,
    MissingScoreContractDigest,
    MissingSourceDigest,
    OutcomeCountOverflow,
    IncompleteActionScenarioCoverage,
    UnknownActionId,
    UnknownScenarioId,
    DuplicateActionScenarioRecord,
    InvalidReplicateCount,
    NonFiniteScoreSample,
    ScoreSampleOutsideDeclaredRange,
    BoundComputationNonFinite,
}

impl fmt::Display for ScoreBoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidFamilywiseAlpha => "familywise alpha must be finite and in (0, 1)",
            Self::InvalidMinimumReplicates => "minimum replicate count must be at least two",
            Self::InvalidScoreRange => "score ranges must be finite, ordered and respect metric domains",
            Self::MissingScoreContractDigest => "score-contract digest must not be empty",
            Self::MissingSourceDigest => "score source digest must not be empty",
            Self::OutcomeCountOverflow => "action × scenario × metric claim count overflowed usize",
            Self::IncompleteActionScenarioCoverage => {
                "exactly one score record is required for every action and scenario pair"
            }
            Self::UnknownActionId => "score record references an unknown action",
            Self::UnknownScenarioId => "score record references an unknown scenario",
            Self::DuplicateActionScenarioRecord => "duplicate score record for an action × scenario pair",
            Self::InvalidReplicateCount => "every metric must have the same sufficient replicate count",
            Self::NonFiniteScoreSample => "score samples must be finite",
            Self::ScoreSampleOutsideDeclaredRange => "score sample lies outside its frozen range",
            Self::BoundComputationNonFinite => "finite-sample score bound was not representable",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ScoreBoundError {}

/// Construct simultaneous one-sided Hoeffding bounds and convert them to
/// ScenarioOutcome records consumed directly by the scenario aggregator.
///
/// With R = action_count × scenario_count records there are six one-sided
/// claims per record: four lower bounds, one upper burden bound, and a second
/// upper utility bound. Each receives delta = familywise_alpha/(6R). Hoeffding
/// is valid for independent bounded Monte Carlo replicates within each series.
/// Bonferroni does not require independence between claims.
pub fn bound_candidate_score_samples(
    action_set: &OperationalActionSet,
    scenario_set: &ScenarioSet,
    samples: &[BoundedScoreSamples],
    config: &ScoreBoundsConfig,
) -> Result<ScoreBoundsReceipt, ScoreBoundError> {
    bound_candidate_score_samples_with_schedule(
        action_set,
        scenario_set,
        samples,
        config,
        BoundSchedule::FixedLook,
    )
}

/// Construct a time-uniform score confidence sequence for every action × scenario.
///
/// For base per-claim alpha delta = familywise_alpha/(6R), the look-n tail
/// probability is delta_n = delta/[n(n+1)]. Since the sum over n >= 1 of
/// 1/[n(n+1)] is one, a union bound over all six claims, all cells and all
/// sample counts gives simultaneous coverage at every look. The radius is
/// range_width * sqrt(log(n(n+1)/delta)/(2n)).
pub fn bound_candidate_score_samples_anytime(
    action_set: &OperationalActionSet,
    scenario_set: &ScenarioSet,
    samples: &[BoundedScoreSamples],
    config: &ScoreBoundsConfig,
) -> Result<ScoreBoundsReceipt, ScoreBoundError> {
    bound_candidate_score_samples_with_schedule(
        action_set,
        scenario_set,
        samples,
        config,
        BoundSchedule::Anytime,
    )
}

fn bound_candidate_score_samples_with_schedule(
    action_set: &OperationalActionSet,
    scenario_set: &ScenarioSet,
    samples: &[BoundedScoreSamples],
    config: &ScoreBoundsConfig,
    schedule: BoundSchedule,
) -> Result<ScoreBoundsReceipt, ScoreBoundError> {
    validate_config(config)?;

    let expected_records = action_set
        .actions()
        .len()
        .checked_mul(scenario_set.scenarios().len())
        .ok_or(ScoreBoundError::OutcomeCountOverflow)?;
    if samples.len() != expected_records {
        return Err(ScoreBoundError::IncompleteActionScenarioCoverage);
    }
    let claim_count = expected_records
        .checked_mul(6)
        .ok_or(ScoreBoundError::OutcomeCountOverflow)?;
    let per_claim_alpha = config.familywise_alpha / claim_count as f64;
    if !per_claim_alpha.is_finite() || per_claim_alpha <= 0.0 {
        return Err(ScoreBoundError::BoundComputationNonFinite);
    }

    let action_ids: BTreeMap<_, _> = action_set
        .actions()
        .iter()
        .map(|action| (action.action_id, ()))
        .collect();
    let scenario_ids: BTreeMap<_, _> = scenario_set
        .scenarios()
        .iter()
        .map(|scenario| (scenario.scenario_id, ()))
        .collect();

    let mut sample_by_pair = BTreeMap::new();
    for record in samples {
        if !action_ids.contains_key(&record.action_id) {
            return Err(ScoreBoundError::UnknownActionId);
        }
        if !scenario_ids.contains_key(&record.scenario_id) {
            return Err(ScoreBoundError::UnknownScenarioId);
        }
        validate_replicate_vectors(record, config)?;
        if sample_by_pair
            .insert((record.action_id, record.scenario_id), record)
            .is_some()
        {
            return Err(ScoreBoundError::DuplicateActionScenarioRecord);
        }
    }
    if sample_by_pair.len() != expected_records {
        return Err(ScoreBoundError::IncompleteActionScenarioCoverage);
    }

    let mut canonical_samples = samples.to_vec();
    canonical_samples.sort_by_key(|record| (record.action_id, record.scenario_id));

    let mut metrics = Vec::with_capacity(expected_records);
    let mut scenario_outcomes = Vec::with_capacity(expected_records);
    for action in action_set.actions() {
        for scenario in scenario_set.scenarios() {
            let record = sample_by_pair
                .get(&(action.action_id, scenario.scenario_id))
                .copied()
                .ok_or(ScoreBoundError::IncompleteActionScenarioCoverage)?;

            let target_information = lower_metric_bound(
                &record.target_information,
                config.ranges.target_information,
                per_claim_alpha,
                schedule,
            )?;
            let model_discrimination_information = lower_metric_bound(
                &record.model_discrimination_information,
                config.ranges.model_discrimination_information,
                per_claim_alpha,
                schedule,
            )?;
            let representativeness = lower_metric_bound(
                &record.representativeness,
                config.ranges.representativeness,
                per_claim_alpha,
                schedule,
            )?;
            let target_utility_lower = lower_metric_bound(
                &record.target_utility,
                config.ranges.target_utility,
                per_claim_alpha,
                schedule,
            )?;
            let target_utility_upper = upper_metric_bound(
                &record.target_utility,
                config.ranges.target_utility,
                per_claim_alpha,
                schedule,
            )?;
            let target_utility = MetricScoreBound {
                point_estimate: target_utility_lower.point_estimate,
                lower_bound: target_utility_lower.lower_bound,
                upper_bound: target_utility_upper.upper_bound,
                sample_count: target_utility_lower.sample_count,
                declared_range: target_utility_lower.declared_range,
            };
            let participant_burden_seconds = upper_metric_bound(
                &record.participant_burden_seconds,
                config.ranges.participant_burden_seconds,
                per_claim_alpha,
                schedule,
            )?;

            metrics.push(ActionScenarioMetricBounds {
                action_id: record.action_id,
                scenario_id: record.scenario_id,
                target_information,
                model_discrimination_information,
                representativeness,
                target_utility,
                participant_burden_seconds,
            });
            scenario_outcomes.push(ScenarioOutcome {
                action_id: record.action_id,
                scenario_id: record.scenario_id,
                operational_mapping_qualified: record.operational_mapping_qualified,
                model_adequacy_qualified: record.model_adequacy_qualified,
                target_information: target_information.point_estimate,
                target_information_lower_bound: target_information.lower_bound,
                model_discrimination_information:
                    model_discrimination_information.point_estimate,
                model_discrimination_information_lower_bound:
                    model_discrimination_information.lower_bound,
                representativeness: representativeness.point_estimate,
                representativeness_lower_bound: representativeness.lower_bound,
                target_utility: target_utility.point_estimate,
                target_utility_lower_bound: target_utility.lower_bound,
                participant_burden_seconds: participant_burden_seconds.point_estimate,
                participant_burden_upper_bound: participant_burden_seconds.upper_bound,
            });
        }
    }

    let (method_id, time_uniform, alpha_spending_rule) = match schedule {
        BoundSchedule::FixedLook => (SCORE_BOUND_METHOD_ID, false, "none"),
        BoundSchedule::Anytime => (
            ANYTIME_SCORE_BOUND_METHOD_ID,
            true,
            ANYTIME_ALPHA_SPENDING_RULE_ID,
        ),
    };

    Ok(ScoreBoundsReceipt {
        method_id,
        time_uniform,
        alpha_spending_rule,
        action_set_digest: action_set.digest().to_owned(),
        action_snapshot: action_set.actions().to_vec(),
        scenario_set_digest: scenario_set.digest().to_owned(),
        scenarios: scenario_set.scenarios().to_vec(),
        familywise_alpha: config.familywise_alpha,
        per_claim_alpha,
        familywise_confidence: 1.0 - config.familywise_alpha,
        claim_count,
        minimum_replicates: config.minimum_replicates,
        ranges: config.ranges,
        score_contract_digest: config.score_contract_digest.clone(),
        source_digest: config.source_digest.clone(),
        root_seed: config.root_seed,
        samples: canonical_samples,
        metrics,
        scenario_outcomes,
    })
}

fn validate_config(config: &ScoreBoundsConfig) -> Result<(), ScoreBoundError> {
    if !config.familywise_alpha.is_finite()
        || config.familywise_alpha <= 0.0
        || config.familywise_alpha >= 1.0
    {
        return Err(ScoreBoundError::InvalidFamilywiseAlpha);
    }
    if config.minimum_replicates < 2 {
        return Err(ScoreBoundError::InvalidMinimumReplicates);
    }
    config.ranges.validate()?;
    if config.score_contract_digest.trim().is_empty() {
        return Err(ScoreBoundError::MissingScoreContractDigest);
    }
    if config.source_digest.trim().is_empty() {
        return Err(ScoreBoundError::MissingSourceDigest);
    }
    Ok(())
}

fn validate_replicate_vectors(
    record: &BoundedScoreSamples,
    config: &ScoreBoundsConfig,
) -> Result<(), ScoreBoundError> {
    let count = record.target_information.len();
    if count < config.minimum_replicates
        || record.model_discrimination_information.len() != count
        || record.representativeness.len() != count
        || record.target_utility.len() != count
        || record.participant_burden_seconds.len() != count
    {
        return Err(ScoreBoundError::InvalidReplicateCount);
    }
    validate_samples(&record.target_information, config.ranges.target_information)?;
    validate_samples(
        &record.model_discrimination_information,
        config.ranges.model_discrimination_information,
    )?;
    validate_samples(&record.representativeness, config.ranges.representativeness)?;
    validate_samples(&record.target_utility, config.ranges.target_utility)?;
    validate_samples(
        &record.participant_burden_seconds,
        config.ranges.participant_burden_seconds,
    )?;
    Ok(())
}

fn validate_samples(values: &[f64], range: ScoreRange) -> Result<(), ScoreBoundError> {
    for value in values {
        if !value.is_finite() {
            return Err(ScoreBoundError::NonFiniteScoreSample);
        }
        if *value < range.lower || *value > range.upper {
            return Err(ScoreBoundError::ScoreSampleOutsideDeclaredRange);
        }
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    let mut average = 0.0;
    for (index, value) in values.iter().enumerate() {
        let n = (index + 1) as f64;
        average += (*value - average) / n;
    }
    average
}

fn hoeffding_radius(
    range: ScoreRange,
    sample_count: usize,
    delta: f64,
    schedule: BoundSchedule,
) -> Result<f64, ScoreBoundError> {
    let width = range.width();
    if width == 0.0 {
        return Ok(0.0);
    }
    let log_inverse_delta = match schedule {
        BoundSchedule::FixedLook => -delta.ln(),
        BoundSchedule::Anytime => {
            let next_count = sample_count
                .checked_add(1)
                .ok_or(ScoreBoundError::BoundComputationNonFinite)?;
            -delta.ln()
                + (sample_count as f64).ln()
                + (next_count as f64).ln()
        }
    };
    let factor = (log_inverse_delta / (2.0 * sample_count as f64)).sqrt();
    if !factor.is_finite() || factor < 0.0 {
        return Err(ScoreBoundError::BoundComputationNonFinite);
    }
    // If factor >= 1, the full declared width is already a conservative
    // radius. This branch also avoids overflowing width * factor.
    Ok(if factor >= 1.0 { width } else { width * factor })
}

fn lower_metric_bound(
    values: &[f64],
    range: ScoreRange,
    delta: f64,
    schedule: BoundSchedule,
) -> Result<MetricScoreBound, ScoreBoundError> {
    let point_estimate = mean(values);
    let radius = hoeffding_radius(range, values.len(), delta, schedule)?;
    let lower_bound = (point_estimate - radius).max(range.lower);
    if !point_estimate.is_finite() || !lower_bound.is_finite() {
        return Err(ScoreBoundError::BoundComputationNonFinite);
    }
    Ok(MetricScoreBound {
        point_estimate,
        lower_bound,
        upper_bound: point_estimate,
        sample_count: values.len(),
        declared_range: range,
    })
}

fn upper_metric_bound(
    values: &[f64],
    range: ScoreRange,
    delta: f64,
    schedule: BoundSchedule,
) -> Result<MetricScoreBound, ScoreBoundError> {
    let point_estimate = mean(values);
    let radius = hoeffding_radius(range, values.len(), delta, schedule)?;
    let upper_bound = (point_estimate + radius).min(range.upper);
    if !point_estimate.is_finite() || !upper_bound.is_finite() {
        return Err(ScoreBoundError::BoundComputationNonFinite);
    }
    Ok(MetricScoreBound {
        point_estimate,
        lower_bound: point_estimate,
        upper_bound,
        sample_count: values.len(),
        declared_range: range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{
        aggregate_scenario_assessments, RobustUtility, ScenarioDefinition, ScenarioSet,
    };
    use crate::sequential::{OperationalAction, StimulusClass};

    fn action_set() -> OperationalActionSet {
        OperationalActionSet::new(
            vec![
                OperationalAction {
                    action_id: 1,
                    audit_stratum: 1,
                    condition_id: 1,
                    stimulus_class: StimulusClass::Noise,
                    stimulus_level: 0.0,
                    response_scale_id: 1,
                    rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 2,
                    audit_stratum: 1,
                    condition_id: 1,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 1.0,
                    response_scale_id: 1,
                    rt_deadline_ms: None,
                },
            ],
            "sha256:action-set",
        )
        .unwrap()
    }

    fn scenario_set() -> ScenarioSet {
        ScenarioSet::new(
            vec![
                ScenarioDefinition { scenario_id: 20, weight: 0.4 },
                ScenarioDefinition { scenario_id: 10, weight: 0.6 },
            ],
            "sha256:scenario-set",
        )
        .unwrap()
    }

    fn ranges() -> ScoreRanges {
        ScoreRanges {
            target_information: ScoreRange { lower: 0.0, upper: 4.0 },
            model_discrimination_information: ScoreRange { lower: 0.0, upper: 3.0 },
            representativeness: ScoreRange { lower: 0.0, upper: 1.0 },
            target_utility: ScoreRange { lower: -10.0, upper: 10.0 },
            participant_burden_seconds: ScoreRange { lower: 0.0, upper: 120.0 },
        }
    }

    fn record(action_id: u32, scenario_id: u16, offset: f64) -> BoundedScoreSamples {
        BoundedScoreSamples {
            action_id,
            scenario_id,
            operational_mapping_qualified: true,
            model_adequacy_qualified: true,
            target_information: vec![1.0 + offset, 1.1 + offset, 0.9 + offset, 1.0 + offset],
            model_discrimination_information: vec![0.5, 0.6, 0.55, 0.55],
            representativeness: vec![0.8, 0.85, 0.75, 0.8],
            target_utility: vec![-2.0 + offset, -1.0 + offset, -1.5 + offset, -1.5 + offset],
            participant_burden_seconds: vec![20.0, 21.0, 19.0, 20.0],
        }
    }

    fn config() -> ScoreBoundsConfig {
        ScoreBoundsConfig {
            familywise_alpha: 0.05,
            minimum_replicates: 4,
            ranges: ranges(),
            score_contract_digest: "sha256:score-contract".to_owned(),
            source_digest: "sha256:dgp-source".to_owned(),
            root_seed: 1234,
        }
    }

    fn records() -> Vec<BoundedScoreSamples> {
        vec![
            record(1, 10, 0.0),
            record(1, 20, 0.1),
            record(2, 10, 0.2),
            record(2, 20, 0.3),
        ]
    }

    #[test]
    fn exact_hoeffding_radius_and_familywise_allocation_are_recorded() {
        let action_set = action_set();
        let scenario_set = scenario_set();
        let receipt = bound_candidate_score_samples(&action_set, &scenario_set, &records(), &config())
            .unwrap();
        assert_eq!(receipt.claim_count, 24);
        assert!((receipt.per_claim_alpha - (0.05 / 24.0)).abs() < 1.0e-12);
        let expected_radius =
            4.0 * ((1.0 / (0.05_f64 / 24.0_f64)).ln() / (2.0 * 4.0)).sqrt();
        let metric = &receipt.metrics[0].target_information;
        assert!((metric.point_estimate - 1.0).abs() < 1.0e-12);
        assert!((metric.lower_bound - (1.0 - expected_radius)).abs() < 1.0e-12);
        assert_eq!(receipt.familywise_confidence, 0.95);
        assert_eq!(receipt.method_id, SCORE_BOUND_METHOD_ID);
    }

    #[test]
    fn negative_utility_and_one_sided_bound_directions_are_supported() {
        let receipt = bound_candidate_score_samples(
            &action_set(),
            &scenario_set(),
            &records(),
            &config(),
        )
        .unwrap();
        let metric = &receipt.metrics[0];
        assert!(metric.target_utility.point_estimate < 0.0);
        assert!(metric.target_utility.lower_bound <= metric.target_utility.point_estimate);
        assert!(metric.target_utility.upper_bound >= metric.target_utility.point_estimate);
        assert!(
            metric.participant_burden_seconds.upper_bound
                >= metric.participant_burden_seconds.point_estimate
        );
        assert!(
            metric.representativeness.lower_bound
                <= metric.representativeness.point_estimate
        );
    }

    #[test]
    fn sample_order_is_canonical_and_receipt_replays_exactly() {
        let action_set = action_set();
        let scenario_set = scenario_set();
        let mut forward = records();
        let first = bound_candidate_score_samples(&action_set, &scenario_set, &forward, &config())
            .unwrap();
        forward.reverse();
        let second = bound_candidate_score_samples(&action_set, &scenario_set, &forward, &config())
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.samples[0].action_id, 1);
        assert_eq!(first.samples[0].scenario_id, 10);
    }

    #[test]
    fn invalid_alpha_ranges_counts_and_sample_domains_fail_closed() {
        let action_set = action_set();
        let scenario_set = scenario_set();
        let mut invalid = config();
        invalid.familywise_alpha = 1.0;
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &records(), &invalid),
            Err(ScoreBoundError::InvalidFamilywiseAlpha)
        );

        invalid = config();
        invalid.minimum_replicates = 1;
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &records(), &invalid),
            Err(ScoreBoundError::InvalidMinimumReplicates)
        );

        invalid = config();
        invalid.ranges.target_utility = ScoreRange { lower: 1.0, upper: -1.0 };
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &records(), &invalid),
            Err(ScoreBoundError::InvalidScoreRange)
        );

        let mut outside = records();
        outside[0].representativeness[0] = 1.1;
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &outside, &config()),
            Err(ScoreBoundError::ScoreSampleOutsideDeclaredRange)
        );

        let mut too_few = records();
        too_few[0].target_information.pop();
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &too_few, &config()),
            Err(ScoreBoundError::InvalidReplicateCount)
        );
    }

    #[test]
    fn nonfinite_samples_and_missing_provenance_reject() {
        let action_set = action_set();
        let scenario_set = scenario_set();
        let mut samples = records();
        samples[0].target_utility[0] = f64::NAN;
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &samples, &config()),
            Err(ScoreBoundError::NonFiniteScoreSample)
        );

        let samples = records();
        let mut invalid = config();
        invalid.score_contract_digest = " ".to_owned();
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &samples, &invalid),
            Err(ScoreBoundError::MissingScoreContractDigest)
        );
        invalid = config();
        invalid.source_digest.clear();
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &samples, &invalid),
            Err(ScoreBoundError::MissingSourceDigest)
        );
    }

    #[test]
    fn missing_duplicate_and_unknown_action_scenario_records_reject() {
        let action_set = action_set();
        let scenario_set = scenario_set();
        let mut missing = records();
        missing.pop();
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &missing, &config()),
            Err(ScoreBoundError::IncompleteActionScenarioCoverage)
        );

        let mut duplicate = records();
        duplicate[1] = duplicate[0].clone();
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &duplicate, &config()),
            Err(ScoreBoundError::DuplicateActionScenarioRecord)
        );

        let mut unknown = records();
        unknown[0].action_id = 99;
        assert_eq!(
            bound_candidate_score_samples(&action_set, &scenario_set, &unknown, &config()),
            Err(ScoreBoundError::UnknownActionId)
        );
    }

    #[test]
    fn output_bounds_feed_scenario_aggregation_without_reconstruction() {
        let action_set = action_set();
        let scenario_set = scenario_set();
        let bounded = bound_candidate_score_samples(
            &action_set,
            &scenario_set,
            &records(),
            &config(),
        )
        .unwrap();
        let aggregated = aggregate_scenario_assessments(
            &action_set,
            &scenario_set,
            &bounded.scenario_outcomes,
            RobustUtility::LowerTailCvar { tail_mass: 0.5 },
            bounded.score_contract_digest.clone(),
        )
        .unwrap();
        assert_eq!(bounded.action_set_digest, action_set.digest());
        assert_eq!(bounded.action_snapshot, action_set.actions());
        assert_eq!(bounded.scenario_set_digest, scenario_set.digest());
        assert_eq!(bounded.scenarios, scenario_set.scenarios());
        assert_eq!(aggregated.assessments.len(), 2);
        assert_eq!(aggregated.scenario_set_digest, scenario_set.digest());
        assert_eq!(aggregated.score_contract_digest, bounded.score_contract_digest);
        for outcome in &bounded.scenario_outcomes {
            assert!(outcome.target_information_lower_bound <= outcome.target_information);
            assert!(
                outcome.model_discrimination_information_lower_bound
                    <= outcome.model_discrimination_information
            );
            assert!(outcome.representativeness_lower_bound <= outcome.representativeness);
            assert!(outcome.target_utility_lower_bound <= outcome.target_utility);
            assert!(outcome.participant_burden_upper_bound >= outcome.participant_burden_seconds);
        }
        for metric in &bounded.metrics {
            assert!(metric.target_utility.lower_bound <= metric.target_utility.point_estimate);
            assert!(metric.target_utility.upper_bound >= metric.target_utility.point_estimate);
        }
    }
}
