// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// B0 deterministic replay oracle with a separate sealed holdout grader.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/replay_oracle.rs -o /tmp/isru-replay-oracle
//   /tmp/isru-replay-oracle

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayManifest {
    pub dataset_id: String,
    pub task_version: String,
    pub oracle_commitment: String,
    pub max_experiments: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateDesign {
    pub candidate_id: String,
    pub parameter_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedOutcome {
    pub candidate_id: String,
    pub result_id: String,
    pub record_digest: String,
    pub utility: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoldoutOutcome {
    pub candidate_id: String,
    pub utility: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayObservation {
    pub request_id: String,
    pub candidate_id: String,
    pub result_id: String,
    pub record_digest: String,
    pub utility: f64,
    pub replay_sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayEvent {
    CandidateMeasured {
        request_id: String,
        candidate_id: String,
        result_id: String,
    },
    CampaignClosed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoldoutPrediction {
    pub candidate_id: String,
    pub predicted_mean: f64,
    pub interval_lower: f64,
    pub interval_upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryScore {
    pub best_measured_utility: Option<f64>,
    pub experiments_completed: u64,
    pub threshold: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearningScore {
    /// Spearman rank correlation between predictions and sealed outcomes.
    /// None means one of the rank vectors has no variation.
    pub spearman_rank_correlation: Option<f64>,
    pub prediction_interval_coverage: f64,
    pub holdout_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    EmptyIdentity(&'static str),
    InvalidUtility(String),
    DuplicateCandidate(String),
    DuplicateResult(String),
    CandidateOutcomeMismatch(String),
    HoldoutOutcomeMismatch(String),
    CandidateOverlap(String),
    UnknownCandidate(String),
    HoldoutCannotBeMeasured(String),
    DuplicateMeasurement(String),
    IdempotencyConflict(String),
    BudgetExhausted,
    CampaignClosed,
    CampaignNotClosed,
    InvalidPrediction(String),
    DuplicatePrediction(String),
    MissingHoldoutPrediction(String),
    UnexpectedHoldoutPrediction(String),
    TooFewHoldoutCandidates,
    InvalidDiscoveryThreshold,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ReplayState {
    closed: bool,
    observed_candidates: BTreeSet<String>,
    seen_requests: BTreeMap<String, ReplayObservation>,
    events: Vec<ReplayEvent>,
}

pub struct ReplayOracle {
    manifest: ReplayManifest,
    measurable_candidates: BTreeMap<String, CandidateDesign>,
    holdout_candidates: BTreeMap<String, CandidateDesign>,
    public_outcomes: BTreeMap<String, RecordedOutcome>,
    sealed_holdout: BTreeMap<String, HoldoutOutcome>,
    state: ReplayState,
}

/// This is the only interface passed to the agent process for B0. It has
/// candidate metadata and visible outcomes only; it has no reference to the
/// sealed holdout labels or the evaluator.
pub struct ReplayAgentApi<'a> {
    manifest: &'a ReplayManifest,
    measurable_candidates: &'a BTreeMap<String, CandidateDesign>,
    holdout_candidates: &'a BTreeMap<String, CandidateDesign>,
    public_outcomes: &'a BTreeMap<String, RecordedOutcome>,
    state: &'a mut ReplayState,
}

impl ReplayOracle {
    pub fn new(
        manifest: ReplayManifest,
        measurable_candidates: Vec<CandidateDesign>,
        public_outcomes: Vec<RecordedOutcome>,
        holdout_candidates: Vec<CandidateDesign>,
        sealed_holdout: Vec<HoldoutOutcome>,
    ) -> Result<Self, ReplayError> {
        for (field, value) in [
            ("dataset_id", manifest.dataset_id.as_str()),
            ("task_version", manifest.task_version.as_str()),
            ("oracle_commitment", manifest.oracle_commitment.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ReplayError::EmptyIdentity(field));
            }
        }
        if manifest.max_experiments == 0 {
            return Err(ReplayError::BudgetExhausted);
        }

        let measurable_candidates = unique_candidates(measurable_candidates)?;
        let holdout_candidates = unique_candidates(holdout_candidates)?;
        let public_outcomes = unique_public_outcomes(public_outcomes)?;
        let sealed_holdout = unique_holdout_outcomes(sealed_holdout)?;

        for candidate_id in measurable_candidates.keys() {
            if !public_outcomes.contains_key(candidate_id) {
                return Err(ReplayError::CandidateOutcomeMismatch(candidate_id.clone()));
            }
            if holdout_candidates.contains_key(candidate_id) {
                return Err(ReplayError::CandidateOverlap(candidate_id.clone()));
            }
        }
        for candidate_id in public_outcomes.keys() {
            if !measurable_candidates.contains_key(candidate_id) {
                return Err(ReplayError::CandidateOutcomeMismatch(candidate_id.clone()));
            }
        }
        for candidate_id in holdout_candidates.keys() {
            if !sealed_holdout.contains_key(candidate_id) {
                return Err(ReplayError::HoldoutOutcomeMismatch(candidate_id.clone()));
            }
            if public_outcomes.contains_key(candidate_id) {
                return Err(ReplayError::CandidateOverlap(candidate_id.clone()));
            }
        }
        for candidate_id in sealed_holdout.keys() {
            if !holdout_candidates.contains_key(candidate_id) {
                return Err(ReplayError::HoldoutOutcomeMismatch(candidate_id.clone()));
            }
        }
        if holdout_candidates.len() < 3 {
            return Err(ReplayError::TooFewHoldoutCandidates);
        }

        Ok(Self {
            manifest,
            measurable_candidates,
            holdout_candidates,
            public_outcomes,
            sealed_holdout,
            state: ReplayState::default(),
        })
    }

    /// Borrow the agent-side interface. The holdout candidate metadata is
    /// visible, but no holdout outcome or grader method is included.
    pub fn agent_api(&mut self) -> ReplayAgentApi<'_> {
        ReplayAgentApi {
            manifest: &self.manifest,
            measurable_candidates: &self.measurable_candidates,
            holdout_candidates: &self.holdout_candidates,
            public_outcomes: &self.public_outcomes,
            state: &mut self.state,
        }
    }

    /// The evaluator is separate from the agent-facing port. Production B0
    /// should run it in a custodian-controlled process after agent closeout.
    pub fn grade_discovery(&self, threshold: f64) -> Result<DiscoveryScore, ReplayError> {
        if !self.state.closed {
            return Err(ReplayError::CampaignNotClosed);
        }
        if !threshold.is_finite() {
            return Err(ReplayError::InvalidDiscoveryThreshold);
        }

        let best = self
            .state
            .seen_requests
            .values()
            .map(|observation| observation.utility)
            .reduce(f64::max);
        let experiments = self.state.observed_candidates.len() as u64;
        Ok(DiscoveryScore {
            best_measured_utility: best,
            experiments_completed: experiments,
            threshold,
            passed: best.is_some_and(|value| value >= threshold),
        })
    }

    /// Return only aggregate metrics; never return hidden holdout labels.
    pub fn grade_learning(
        &self,
        predictions: Vec<HoldoutPrediction>,
    ) -> Result<LearningScore, ReplayError> {
        if !self.state.closed {
            return Err(ReplayError::CampaignNotClosed);
        }

        let mut by_id = BTreeMap::new();
        for prediction in predictions {
            if prediction.candidate_id.trim().is_empty() {
                return Err(ReplayError::EmptyIdentity("prediction.candidate_id"));
            }
            if !prediction.predicted_mean.is_finite()
                || !prediction.interval_lower.is_finite()
                || !prediction.interval_upper.is_finite()
                || prediction.interval_lower > prediction.interval_upper
            {
                return Err(ReplayError::InvalidPrediction(prediction.candidate_id));
            }
            if !self
                .holdout_candidates
                .contains_key(&prediction.candidate_id)
            {
                return Err(ReplayError::UnexpectedHoldoutPrediction(
                    prediction.candidate_id,
                ));
            }
            let candidate_id = prediction.candidate_id.clone();
            if by_id.insert(candidate_id.clone(), prediction).is_some() {
                return Err(ReplayError::DuplicatePrediction(candidate_id));
            }
        }

        for candidate_id in self.holdout_candidates.keys() {
            if !by_id.contains_key(candidate_id) {
                return Err(ReplayError::MissingHoldoutPrediction(candidate_id.clone()));
            }
        }

        let mut predicted_values = Vec::with_capacity(self.holdout_candidates.len());
        let mut actual_values = Vec::with_capacity(self.holdout_candidates.len());
        let mut covered = 0_usize;

        for candidate_id in self.holdout_candidates.keys() {
            let prediction = by_id
                .get(candidate_id)
                .expect("complete holdout predictions checked above");
            let actual = self
                .sealed_holdout
                .get(candidate_id)
                .expect("holdout manifest consistency checked at construction")
                .utility;
            predicted_values.push(prediction.predicted_mean);
            actual_values.push(actual);
            if actual >= prediction.interval_lower && actual <= prediction.interval_upper {
                covered += 1;
            }
        }

        Ok(LearningScore {
            spearman_rank_correlation: spearman(&predicted_values, &actual_values),
            prediction_interval_coverage: covered as f64 / actual_values.len() as f64,
            holdout_count: actual_values.len(),
        })
    }

    pub fn events(&self) -> &[ReplayEvent] {
        &self.state.events
    }
}

impl ReplayAgentApi<'_> {
    pub fn manifest(&self) -> &ReplayManifest {
        self.manifest
    }

    pub fn measurable_catalog(&self) -> Vec<&CandidateDesign> {
        self.measurable_candidates.values().collect()
    }

    /// Candidate metadata only: no outcome labels are exposed through this
    /// method, and these candidates are forbidden from measurement.
    pub fn holdout_catalog(&self) -> Vec<&CandidateDesign> {
        self.holdout_candidates.values().collect()
    }

    /// Repeated identical request IDs are idempotent. A new request for an
    /// already measured candidate is rejected; B0 v0 intentionally has no
    /// replicate semantics until a replicate-aware dataset format exists.
    pub fn measure(
        &mut self,
        request_id: &str,
        candidate_id: &str,
    ) -> Result<ReplayObservation, ReplayError> {
        if request_id.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("request_id"));
        }
        if candidate_id.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("candidate_id"));
        }
        if self.state.closed {
            return Err(ReplayError::CampaignClosed);
        }

        if let Some(previous) = self.state.seen_requests.get(request_id) {
            if previous.candidate_id == candidate_id {
                return Ok(previous.clone());
            }
            return Err(ReplayError::IdempotencyConflict(request_id.to_string()));
        }

        if self.holdout_candidates.contains_key(candidate_id) {
            return Err(ReplayError::HoldoutCannotBeMeasured(
                candidate_id.to_string(),
            ));
        }
        if !self.measurable_candidates.contains_key(candidate_id) {
            return Err(ReplayError::UnknownCandidate(candidate_id.to_string()));
        }
        if self.state.observed_candidates.len() as u64 >= self.manifest.max_experiments {
            return Err(ReplayError::BudgetExhausted);
        }
        if self.state.observed_candidates.contains(candidate_id) {
            return Err(ReplayError::DuplicateMeasurement(candidate_id.to_string()));
        }

        let outcome = self
            .public_outcomes
            .get(candidate_id)
            .expect("public candidate/outcome consistency checked at construction");
        let observation = ReplayObservation {
            request_id: request_id.to_string(),
            candidate_id: candidate_id.to_string(),
            result_id: outcome.result_id.clone(),
            record_digest: outcome.record_digest.clone(),
            utility: outcome.utility,
            replay_sequence: self.state.events.len() as u64,
        };

        self.state
            .observed_candidates
            .insert(candidate_id.to_string());
        self.state
            .seen_requests
            .insert(request_id.to_string(), observation.clone());
        self.state.events.push(ReplayEvent::CandidateMeasured {
            request_id: request_id.to_string(),
            candidate_id: candidate_id.to_string(),
            result_id: outcome.result_id.clone(),
        });

        Ok(observation)
    }

    pub fn close_campaign(&mut self) -> Result<(), ReplayError> {
        if self.state.closed {
            return Err(ReplayError::CampaignClosed);
        }
        self.state.closed = true;
        self.state.events.push(ReplayEvent::CampaignClosed);
        Ok(())
    }
}

fn unique_candidates(
    candidates: Vec<CandidateDesign>,
) -> Result<BTreeMap<String, CandidateDesign>, ReplayError> {
    let mut result = BTreeMap::new();
    for candidate in candidates {
        if candidate.candidate_id.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("candidate_id"));
        }
        if candidate.parameter_digest.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("parameter_digest"));
        }
        let id = candidate.candidate_id.clone();
        if result.insert(id.clone(), candidate).is_some() {
            return Err(ReplayError::DuplicateCandidate(id));
        }
    }
    Ok(result)
}

fn unique_public_outcomes(
    outcomes: Vec<RecordedOutcome>,
) -> Result<BTreeMap<String, RecordedOutcome>, ReplayError> {
    let mut result = BTreeMap::new();
    let mut result_ids = BTreeSet::new();
    for outcome in outcomes {
        if outcome.candidate_id.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("outcome.candidate_id"));
        }
        if outcome.result_id.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("outcome.result_id"));
        }
        if outcome.record_digest.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("outcome.record_digest"));
        }
        if !outcome.utility.is_finite() {
            return Err(ReplayError::InvalidUtility(outcome.candidate_id));
        }
        if !result_ids.insert(outcome.result_id.clone()) {
            return Err(ReplayError::DuplicateResult(outcome.result_id));
        }
        let id = outcome.candidate_id.clone();
        if result.insert(id.clone(), outcome).is_some() {
            return Err(ReplayError::DuplicateCandidate(id));
        }
    }
    Ok(result)
}

fn unique_holdout_outcomes(
    outcomes: Vec<HoldoutOutcome>,
) -> Result<BTreeMap<String, HoldoutOutcome>, ReplayError> {
    let mut result = BTreeMap::new();
    for outcome in outcomes {
        if outcome.candidate_id.trim().is_empty() {
            return Err(ReplayError::EmptyIdentity("holdout.candidate_id"));
        }
        if !outcome.utility.is_finite() {
            return Err(ReplayError::InvalidUtility(outcome.candidate_id));
        }
        let id = outcome.candidate_id.clone();
        if result.insert(id.clone(), outcome).is_some() {
            return Err(ReplayError::DuplicateCandidate(id));
        }
    }
    Ok(result)
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));

    let mut output = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        for index in &order[start..end] {
            output[*index] = average_rank;
        }
        start = end;
    }
    output
}

fn spearman(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 3 {
        return None;
    }
    let left_ranks = ranks(left);
    let right_ranks = ranks(right);
    pearson(&left_ranks, &right_ranks)
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;

    for (a, b) in left.iter().zip(right) {
        let x = *a - left_mean;
        let y = *b - right_mean;
        covariance += x * y;
        left_variance += x * x;
        right_variance += y * y;
    }

    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some(covariance / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str) -> CandidateDesign {
        CandidateDesign {
            candidate_id: id.to_string(),
            parameter_digest: format!("sha256:params-{id}"),
        }
    }

    fn public_outcome(id: &str, utility: f64) -> RecordedOutcome {
        RecordedOutcome {
            candidate_id: id.to_string(),
            result_id: format!("result-{id}"),
            record_digest: format!("sha256:record-{id}"),
            utility,
        }
    }

    fn holdout_outcome(id: &str, utility: f64) -> HoldoutOutcome {
        HoldoutOutcome {
            candidate_id: id.to_string(),
            utility,
        }
    }

    fn oracle() -> ReplayOracle {
        ReplayOracle::new(
            ReplayManifest {
                dataset_id: "optical-isru-dev-v0".to_string(),
                task_version: "0.1.0".to_string(),
                oracle_commitment: "sha256:sealed-outcomes-commitment".to_string(),
                max_experiments: 2,
            },
            vec![candidate("train-a"), candidate("train-b")],
            vec![
                public_outcome("train-a", 0.25),
                public_outcome("train-b", 0.85),
            ],
            vec![
                candidate("holdout-a"),
                candidate("holdout-b"),
                candidate("holdout-c"),
            ],
            vec![
                holdout_outcome("holdout-a", 0.2),
                holdout_outcome("holdout-b", 0.6),
                holdout_outcome("holdout-c", 0.9),
            ],
        )
        .unwrap()
    }

    #[test]
    fn agent_port_exposes_candidate_metadata_but_only_measurement_reveals_public_outcome() {
        let mut value = oracle();
        let mut agent = value.agent_api();
        assert_eq!(agent.measurable_catalog().len(), 2);
        assert_eq!(agent.holdout_catalog().len(), 3);
        let observation = agent.measure("request-1", "train-a").unwrap();
        assert_eq!(observation.utility, 0.25);
        assert!(matches!(
            agent.measure("request-2", "holdout-a"),
            Err(ReplayError::HoldoutCannotBeMeasured(_))
        ));
    }

    #[test]
    fn identical_request_is_idempotent_and_does_not_spend_extra_budget() {
        let mut value = oracle();
        let mut agent = value.agent_api();
        let first = agent.measure("request-1", "train-a").unwrap();
        let second = agent.measure("request-1", "train-a").unwrap();
        assert_eq!(first, second);
        assert_eq!(first.replay_sequence, 0);
        assert_eq!(agent.state.observed_candidates.len(), 1);
    }

    #[test]
    fn request_id_reuse_for_different_candidate_is_rejected() {
        let mut value = oracle();
        let mut agent = value.agent_api();
        agent.measure("request-1", "train-a").unwrap();
        assert_eq!(
            agent.measure("request-1", "train-b"),
            Err(ReplayError::IdempotencyConflict("request-1".to_string()))
        );
    }

    #[test]
    fn candidate_cannot_be_measured_twice_without_explicit_replicate_contract() {
        let mut value = oracle();
        let mut agent = value.agent_api();
        agent.measure("request-1", "train-a").unwrap();
        assert_eq!(
            agent.measure("request-2", "train-a"),
            Err(ReplayError::DuplicateMeasurement("train-a".to_string()))
        );
    }

    #[test]
    fn experiment_budget_is_a_hard_limit() {
        let mut value = oracle();
        let mut agent = value.agent_api();
        agent.measure("request-1", "train-a").unwrap();
        agent.measure("request-2", "train-b").unwrap();
        assert_eq!(
            agent.measure("request-3", "train-a"),
            Err(ReplayError::BudgetExhausted)
        );
    }

    #[test]
    fn closeout_blocks_later_measurement() {
        let mut value = oracle();
        {
            let mut agent = value.agent_api();
            agent.close_campaign().unwrap();
            assert_eq!(
                agent.measure("request-1", "train-a"),
                Err(ReplayError::CampaignClosed)
            );
        }
        assert!(matches!(
            value.grade_discovery(0.5),
            Ok(DiscoveryScore {
                passed: false,
                experiments_completed: 0,
                ..
            })
        ));
    }

    #[test]
    fn discovery_gate_scores_only_selected_recorded_outcomes() {
        let mut value = oracle();
        {
            let mut agent = value.agent_api();
            agent.measure("request-1", "train-a").unwrap();
            agent.measure("request-2", "train-b").unwrap();
            agent.close_campaign().unwrap();
        }
        let score = value.grade_discovery(0.8).unwrap();
        assert_eq!(score.experiments_completed, 2);
        assert_eq!(score.best_measured_utility, Some(0.85));
        assert!(score.passed);
    }

    #[test]
    fn learning_gate_returns_aggregate_metrics_without_releasing_holdout_labels() {
        let mut value = oracle();
        value.agent_api().close_campaign().unwrap();
        let score = value
            .grade_learning(vec![
                HoldoutPrediction {
                    candidate_id: "holdout-a".to_string(),
                    predicted_mean: 0.1,
                    interval_lower: 0.0,
                    interval_upper: 0.3,
                },
                HoldoutPrediction {
                    candidate_id: "holdout-b".to_string(),
                    predicted_mean: 0.7,
                    interval_lower: 0.5,
                    interval_upper: 0.8,
                },
                HoldoutPrediction {
                    candidate_id: "holdout-c".to_string(),
                    predicted_mean: 0.95,
                    interval_lower: 0.8,
                    interval_upper: 1.0,
                },
            ])
            .unwrap();
        assert_eq!(score.spearman_rank_correlation, Some(1.0));
        assert_eq!(score.prediction_interval_coverage, 1.0);
        assert_eq!(score.holdout_count, 3);
    }

    #[test]
    fn missing_holdout_prediction_does_not_silently_shrink_the_test_set() {
        let mut value = oracle();
        value.agent_api().close_campaign().unwrap();
        assert_eq!(
            value.grade_learning(vec![HoldoutPrediction {
                candidate_id: "holdout-a".to_string(),
                predicted_mean: 0.1,
                interval_lower: 0.0,
                interval_upper: 0.3,
            }]),
            Err(ReplayError::MissingHoldoutPrediction(
                "holdout-b".to_string()
            ))
        );
    }

    #[test]
    fn duplicate_and_unexpected_holdout_predictions_are_rejected() {
        let mut value = oracle();
        value.agent_api().close_campaign().unwrap();
        let prediction = HoldoutPrediction {
            candidate_id: "holdout-a".to_string(),
            predicted_mean: 0.1,
            interval_lower: 0.0,
            interval_upper: 0.3,
        };
        assert_eq!(
            value.grade_learning(vec![prediction.clone(), prediction]),
            Err(ReplayError::DuplicatePrediction("holdout-a".to_string()))
        );
        assert!(matches!(
            value.grade_learning(vec![HoldoutPrediction {
                candidate_id: "unknown".to_string(),
                predicted_mean: 0.1,
                interval_lower: 0.0,
                interval_upper: 0.3,
            }]),
            Err(ReplayError::UnexpectedHoldoutPrediction(_))
        ));
    }

    #[test]
    fn grader_cannot_run_before_campaign_closeout() {
        let value = oracle();
        assert_eq!(
            value.grade_discovery(0.5),
            Err(ReplayError::CampaignNotClosed)
        );
        assert_eq!(
            value.grade_learning(Vec::new()),
            Err(ReplayError::CampaignNotClosed)
        );
    }

    #[test]
    fn constructor_rejects_overlapping_or_incomplete_public_and_holdout_sets() {
        let manifest = ReplayManifest {
            dataset_id: "d".to_string(),
            task_version: "v1".to_string(),
            oracle_commitment: "commit".to_string(),
            max_experiments: 1,
        };
        let overlap = ReplayOracle::new(
            manifest.clone(),
            vec![candidate("same")],
            vec![public_outcome("same", 0.5)],
            vec![candidate("same"), candidate("h2"), candidate("h3")],
            vec![
                holdout_outcome("same", 0.4),
                holdout_outcome("h2", 0.5),
                holdout_outcome("h3", 0.6),
            ],
        );
        assert_eq!(
            overlap.err(),
            Some(ReplayError::CandidateOverlap("same".to_string()))
        );

        let missing = ReplayOracle::new(
            manifest,
            vec![candidate("train")],
            Vec::new(),
            vec![candidate("h1"), candidate("h2"), candidate("h3")],
            vec![
                holdout_outcome("h1", 0.1),
                holdout_outcome("h2", 0.2),
                holdout_outcome("h3", 0.3),
            ],
        );
        assert_eq!(
            missing.err(),
            Some(ReplayError::CandidateOutcomeMismatch("train".to_string()))
        );
    }

    #[test]
    fn rank_correlation_is_none_for_constant_predictions() {
        assert_eq!(spearman(&[1.0, 1.0, 1.0], &[0.1, 0.2, 0.3]), None);
    }

    #[test]
    fn non_finite_outcomes_and_invalid_intervals_are_rejected() {
        let bad = unique_public_outcomes(vec![public_outcome("bad", f64::NAN)]);
        assert_eq!(
            bad.err(),
            Some(ReplayError::InvalidUtility("bad".to_string()))
        );

        let mut value = oracle();
        value.agent_api().close_campaign().unwrap();
        assert_eq!(
            value.grade_learning(vec![HoldoutPrediction {
                candidate_id: "holdout-a".to_string(),
                predicted_mean: 0.2,
                interval_lower: 0.4,
                interval_upper: 0.3,
            }]),
            Err(ReplayError::InvalidPrediction("holdout-a".to_string()))
        );
    }

    #[test]
    fn replay_events_have_deterministic_sequences_and_closeout() {
        let mut value = oracle();
        {
            let mut agent = value.agent_api();
            agent.measure("request-1", "train-a").unwrap();
            agent.measure("request-2", "train-b").unwrap();
            agent.close_campaign().unwrap();
        }
        assert!(matches!(
            value.events(),
            [
                ReplayEvent::CandidateMeasured { .. },
                ReplayEvent::CandidateMeasured { .. },
                ReplayEvent::CampaignClosed
            ]
        ));
    }
}
