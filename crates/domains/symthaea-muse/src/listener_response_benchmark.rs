// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic temporal-holdout baselines for listener-response research.
//!
//! This evaluates reported pre/post change, not objective emotion and not
//! causal effect. A global collection ordinal makes the cutoff explicit and
//! prevents future sessions from contributing to training statistics.

use crate::listener_response::{ListenerResponseIssue, ListenerResponseRecord, SelfReportedAffect};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const LISTENER_RESPONSE_BENCHMARK_VERSION: &str = "muse-listener-response-benchmark-v2";
pub const PARTICIPANT_BOOTSTRAP_REPLICATES: usize = 2_000;
pub const MIN_PARTICIPANTS_FOR_BOOTSTRAP: usize = 10;
pub const DEFAULT_PERSONAL_PRIOR_STRENGTH: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalHoldoutSplit {
    /// Every record at or before this global collection ordinal is training data.
    pub training_through_sequence: u64,
    /// Every record at or after this ordinal is test data; the middle is embargoed.
    pub testing_from_sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseChannel {
    Valence,
    Arousal,
    Tension,
    Energy,
}

impl ResponseChannel {
    pub const ALL: [Self; 4] = [Self::Valence, Self::Arousal, Self::Tension, Self::Energy];

    fn value(self, affect: SelfReportedAffect) -> Option<f32> {
        match self {
            Self::Valence => affect.valence,
            Self::Arousal => affect.arousal,
            Self::Tension => affect.tension,
            Self::Energy => affect.energy,
        }
    }
}

/// Mean absolute error on the reported change (after minus before).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineComparator {
    NoChange,
    PopulationDelta,
}

/// Equal-participant-weighted improvement of personal-history prediction over a
/// comparator, restricted to the exact same eligible test episodes.
/// Positive improvement means the personal-history predictor has lower MAE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticipantClusterComparison {
    pub comparator: BaselineComparator,
    pub participants: usize,
    pub matched_test_observations: usize,
    pub mean_participant_mae_improvement: Option<f64>,
    pub median_participant_mae_improvement: Option<f64>,
    pub participants_improved_fraction: Option<f64>,
    /// Percentile interval from participant-resampled cluster bootstrap.
    /// Absent below MIN_PARTICIPANTS_FOR_BOOTSTRAP eligible participants.
    pub bootstrap_confidence_interval_95: Option<[f64; 2]>,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
}

#[derive(Debug, Default)]
struct ParticipantErrors {
    no_change: Vec<f64>,
    population_delta: Vec<f64>,
    personal_history: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelBenchmarkMetrics {
    pub channel: ResponseChannel,
    pub scored_test_observations: usize,
    pub population_training_mean_delta: Option<f64>,
    pub no_change_mae: Option<f64>,
    pub population_delta_mae: Option<f64>,
    /// Matched-subset errors for a fair comparison with personal_history_mae.
    pub no_change_mae_on_personal_history: Option<f64>,
    pub population_delta_mae_on_personal_history: Option<f64>,
    /// Only test episodes with at least one earlier session from this listener.
    pub personal_history_mae: Option<f64>,
    /// Population mean shrunk toward personal history; no-change is the fallback
    /// only when no training target exists for this channel.
    pub shrunk_personal_mae: Option<f64>,
    pub personal_history_observations: usize,
    pub personal_history_coverage: f64,
    /// Participant-clustered, equal-participant-weighted comparison on matched episodes.
    pub participant_cluster_comparisons: Vec<ParticipantClusterComparison>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListenerResponseBenchmarkReport {
    pub benchmark_version: String,
    pub split: TemporalHoldoutSplit,
    pub training_records: usize,
    pub embargoed_records: usize,
    pub testing_records: usize,
    /// Test episodes with no channel having both pre- and post-report values.
    pub target_missing_records: usize,
    pub prior_strength: f64,
    pub channels: Vec<ChannelBenchmarkMetrics>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkIssue {
    NoRecords,
    InvalidSplit,
    InvalidRecord { index: usize, issue: ListenerResponseIssue },
    DuplicateCollectionSequence { sequence: u64 },
    DuplicateEpisodeId { episode_id: String },
    ParticipantChronologyViolation { participant_token: String, sequence: u64 },
    SessionCrossesSplitBoundary { participant_token: String, session_index: u32 },
    EmptyTrainingPartition,
    EmptyTestingPartition,
}

/// Evaluate simple, auditable baselines on a global future holdout.
///
/// Population statistics use training records only. A personal estimate uses
/// that listener's earlier sessions; another presentation in the same session
/// cannot leak into their personal history. prior_strength is a pseudo-count
/// for shrinkage toward the population mean.
pub fn evaluate_temporal_holdout(
    records: &[ListenerResponseRecord],
    split: TemporalHoldoutSplit,
    prior_strength: f64,
) -> Result<ListenerResponseBenchmarkReport, Vec<BenchmarkIssue>> {
    if records.is_empty() {
        return Err(vec![BenchmarkIssue::NoRecords]);
    }

    let mut issues = Vec::new();
    if split.training_through_sequence == 0
        || split.testing_from_sequence <= split.training_through_sequence
        || !prior_strength.is_finite()
        || prior_strength < 0.0
    {
        issues.push(BenchmarkIssue::InvalidSplit);
    }

    let mut sequences = BTreeSet::new();
    let mut episodes = BTreeSet::new();
    let mut by_participant: BTreeMap<String, Vec<&ListenerResponseRecord>> = BTreeMap::new();
    let mut session_partitions: BTreeMap<(String, u32), BTreeSet<u8>> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        for issue in record.validate() {
            issues.push(BenchmarkIssue::InvalidRecord { index, issue });
        }
        if !sequences.insert(record.collection_sequence) {
            issues.push(BenchmarkIssue::DuplicateCollectionSequence {
                sequence: record.collection_sequence,
            });
        }
        if !episodes.insert(record.episode_id.as_str()) {
            issues.push(BenchmarkIssue::DuplicateEpisodeId {
                episode_id: record.episode_id.clone(),
            });
        }
        by_participant
            .entry(record.participant_token.clone())
            .or_default()
            .push(record);
        let partition = if record.collection_sequence <= split.training_through_sequence {
            0
        } else if record.collection_sequence < split.testing_from_sequence {
            1
        } else {
            2
        };
        session_partitions
            .entry((record.participant_token.clone(), record.session_index))
            .or_default()
            .insert(partition);
    }

    for ((participant_token, session_index), partitions) in session_partitions {
        if partitions.len() > 1 {
            issues.push(BenchmarkIssue::SessionCrossesSplitBoundary {
                participant_token,
                session_index,
            });
        }
    }

    for (participant, participant_records) in &mut by_participant {
        participant_records.sort_by_key(|record| record.collection_sequence);
        let mut prior_session = 0;
        let mut prior_order = 0;
        for record in participant_records.iter() {
            if record.session_index < prior_session
                || (record.session_index == prior_session
                    && record.order_in_session <= prior_order)
            {
                issues.push(BenchmarkIssue::ParticipantChronologyViolation {
                    participant_token: participant.as_str().to_owned(),
                    sequence: record.collection_sequence,
                });
            }
            if record.session_index != prior_session {
                prior_session = record.session_index;
                prior_order = record.order_in_session;
            } else {
                prior_order = record.order_in_session;
            }
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let training: Vec<&ListenerResponseRecord> = records
        .iter()
        .filter(|record| record.collection_sequence <= split.training_through_sequence)
        .collect();
    let embargoed = records
        .iter()
        .filter(|record| {
            record.collection_sequence > split.training_through_sequence
                && record.collection_sequence < split.testing_from_sequence
        })
        .count();
    let testing: Vec<&ListenerResponseRecord> = records
        .iter()
        .filter(|record| record.collection_sequence >= split.testing_from_sequence)
        .collect();

    if training.is_empty() {
        issues.push(BenchmarkIssue::EmptyTrainingPartition);
    }
    if testing.is_empty() {
        issues.push(BenchmarkIssue::EmptyTestingPartition);
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let target_missing_records = testing
        .iter()
        .filter(|record| {
            ResponseChannel::ALL
                .into_iter()
                .all(|channel| reported_delta(record, channel).is_none())
        })
        .count();

    let mut channels = Vec::with_capacity(ResponseChannel::ALL.len());
    for channel in ResponseChannel::ALL {
        let training_deltas: Vec<f64> = training
            .iter()
            .filter_map(|record| reported_delta(record, channel))
            .collect();
        let population_mean = mean(&training_deltas);

        let mut no_change_errors = Vec::new();
        let mut population_errors = Vec::new();
        let mut personal_errors = Vec::new();
        let mut no_change_on_personal_errors = Vec::new();
        let mut population_on_personal_errors = Vec::new();
        let mut shrinkage_errors = Vec::new();
        let mut personal_history_observations = 0usize;
        let mut scored_test_observations = 0usize;
        let mut participant_errors: BTreeMap<String, ParticipantErrors> = BTreeMap::new();

        for test_record in &testing {
            let Some(actual_delta) = reported_delta(test_record, channel) else {
                continue;
            };
            scored_test_observations += 1;
            no_change_errors.push(actual_delta.abs());

            if let Some(population_prior) = population_mean {
                population_errors.push((actual_delta - population_prior).abs());
            }

            let history: Vec<f64> = training
                .iter()
                .filter(|train_record| {
                    train_record.participant_token == test_record.participant_token
                        && train_record.session_index < test_record.session_index
                        && train_record.collection_sequence < test_record.collection_sequence
                })
                .filter_map(|record| reported_delta(record, channel))
                .collect();

            let prior = population_mean.unwrap_or(0.0);
            let prediction = match mean(&history) {
                Some(personal_mean) => {
                    personal_history_observations += 1;
                    let personal_error = (actual_delta - personal_mean).abs();
                    let no_change_error = actual_delta.abs();
                    personal_errors.push(personal_error);
                    no_change_on_personal_errors.push(no_change_error);
                    let errors = participant_errors
                        .entry(test_record.participant_token.clone())
                        .or_default();
                    errors.personal_history.push(personal_error);
                    errors.no_change.push(no_change_error);
                    if let Some(population_prior) = population_mean {
                        let population_error = (actual_delta - population_prior).abs();
                        population_on_personal_errors.push(population_error);
                        errors.population_delta.push(population_error);
                    }
                    let weight =
                        history.len() as f64 / (history.len() as f64 + prior_strength);
                    prior + weight * (personal_mean - prior)
                }
                None => prior,
            };
            shrinkage_errors.push((actual_delta - prediction).abs());
        }

        let participant_cluster_comparisons = vec![
            participant_cluster_comparison(
                &participant_errors,
                BaselineComparator::NoChange,
                0x4d4c_5242_u64 ^ channel_seed(channel) ^ 0x01,
            ),
            participant_cluster_comparison(
                &participant_errors,
                BaselineComparator::PopulationDelta,
                0x4d4c_5242_u64 ^ channel_seed(channel) ^ 0x02,
            ),
        ];
        channels.push(ChannelBenchmarkMetrics {
            channel,
            scored_test_observations,
            population_training_mean_delta: population_mean,
            no_change_mae: mean(&no_change_errors),
            population_delta_mae: mean(&population_errors),
            no_change_mae_on_personal_history: mean(&no_change_on_personal_errors),
            population_delta_mae_on_personal_history: mean(&population_on_personal_errors),
            personal_history_mae: mean(&personal_errors),
            shrunk_personal_mae: mean(&shrinkage_errors),
            personal_history_observations,
            personal_history_coverage: if scored_test_observations == 0 {
                0.0
            } else {
                personal_history_observations as f64 / scored_test_observations as f64
            },
            participant_cluster_comparisons,
        });
    }

    Ok(ListenerResponseBenchmarkReport {
        benchmark_version: LISTENER_RESPONSE_BENCHMARK_VERSION.to_owned(),
        split,
        training_records: training.len(),
        embargoed_records: embargoed,
        testing_records: testing.len(),
        target_missing_records,
        prior_strength,
        channels,
    })
}

fn channel_seed(channel: ResponseChannel) -> u64 {
    match channel {
        ResponseChannel::Valence => 0x11,
        ResponseChannel::Arousal => 0x22,
        ResponseChannel::Tension => 0x33,
        ResponseChannel::Energy => 0x44,
    }
}

fn participant_cluster_comparison(
    grouped: &BTreeMap<String, ParticipantErrors>,
    comparator: BaselineComparator,
    seed: u64,
) -> ParticipantClusterComparison {
    let mut improvements = Vec::new();
    let mut matched_observations = 0usize;
    for errors in grouped.values() {
        let baseline = match comparator {
            BaselineComparator::NoChange => &errors.no_change,
            BaselineComparator::PopulationDelta => &errors.population_delta,
        };
        let common = baseline.len().min(errors.personal_history.len());
        if common == 0 {
            continue;
        }
        let baseline_mae = mean(&baseline[..common]).expect("non-empty matched baseline");
        let personal_mae =
            mean(&errors.personal_history[..common]).expect("non-empty personal history");
        improvements.push(baseline_mae - personal_mae);
        matched_observations += common;
    }

    let mean_improvement = mean(&improvements);
    let mut ordered = improvements.clone();
    ordered.sort_by(f64::total_cmp);
    let median_improvement = median_sorted(&ordered);
    let improved_fraction = if improvements.is_empty() {
        None
    } else {
        Some(
            improvements.iter().filter(|&&effect| effect > 0.0).count() as f64
                / improvements.len() as f64,
        )
    };
    let (bootstrap_replicates, confidence_interval) =
        if improvements.len() >= MIN_PARTICIPANTS_FOR_BOOTSTRAP {
        (
            PARTICIPANT_BOOTSTRAP_REPLICATES,
            Some(participant_bootstrap_ci(
                &improvements,
                PARTICIPANT_BOOTSTRAP_REPLICATES,
                seed,
            )),
        )
    } else {
        (0, None)
    };

    ParticipantClusterComparison {
        comparator,
        participants: improvements.len(),
        matched_test_observations: matched_observations,
        mean_participant_mae_improvement: mean_improvement,
        median_participant_mae_improvement: median_improvement,
        participants_improved_fraction: improved_fraction,
        bootstrap_confidence_interval_95: confidence_interval,
        bootstrap_replicates,
        bootstrap_seed: seed,
    }
}

fn median_sorted(sorted: &[f64]) -> Option<f64> {
    match sorted.len() {
        0 => None,
        n if n % 2 == 1 => Some(sorted[n / 2]),
        n => Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0),
    }
}

fn participant_bootstrap_ci(values: &[f64], replicates: usize, seed: u64) -> [f64; 2] {
    debug_assert!(values.len() >= MIN_PARTICIPANTS_FOR_BOOTSTRAP);
    debug_assert!(replicates > 0);
    let mut rng = SplitMix64::new(seed);
    let mut draws = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.next_index(values.len())];
        }
        draws.push(total / values.len() as f64);
    }
    draws.sort_by(f64::total_cmp);
    [
        percentile_sorted(&draws, 0.025),
        percentile_sorted(&draws, 0.975),
    ]
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}

/// Small fixed SplitMix64 generator keeps resampling reproducible without
/// relying on platform RNG defaults or external state.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

fn reported_delta(record: &ListenerResponseRecord, channel: ResponseChannel) -> Option<f64> {
    let before = record.felt_before.and_then(|affect| channel.value(affect))?;
    let after = record.felt_after.and_then(|affect| channel.value(affect))?;
    Some(after as f64 - before as f64)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listener_response::{
        LISTENER_RESPONSE_SCHEMA_VERSION, ListeningIntent, SelfReportedAffect,
    };

    fn episode(
        id: &str,
        participant: &str,
        session: u32,
        order: u32,
        sequence: u64,
        valence_before: f32,
        valence_after: f32,
    ) -> ListenerResponseRecord {
        ListenerResponseRecord {
            schema_version: LISTENER_RESPONSE_SCHEMA_VERSION.to_owned(),
            episode_id: id.to_owned(),
            participant_token: participant.to_owned(),
            presentation_token: format!("presentation-{}", id),
            audio_sha256: format!("{:064x}", sequence),
            score_sha256: None,
            model_version: Some("fixture-model-v1".to_owned()),
            collection_sequence: sequence,
            session_index: session,
            order_in_session: order,
            intent: ListeningIntent::Enjoy,
            target_affect: None,
            activity: None,
            familiarity_1_to_5: None,
            felt_before: Some(SelfReportedAffect {
                valence: Some(valence_before),
                arousal: Some(0.4),
                tension: Some(0.3),
                energy: Some(0.4),
            }),
            during: Vec::new(),
            felt_after: Some(SelfReportedAffect {
                valence: Some(valence_after),
                arousal: Some(0.4),
                tension: Some(0.3),
                energy: Some(0.4),
            }),
            perceived_music: None,
            enjoyment_1_to_7: Some(5),
            goal_attainment_1_to_7: None,
            playback_completed: true,
            consent_confirmed: true,
            consent_protocol_version: "synthetic-fixture-consent-v1".to_owned(),
        }
    }

    #[test]
    fn temporal_holdout_reports_auditable_baselines() {
        let records = vec![
            episode("a", "p1", 1, 1, 1, 0.0, 0.4),
            episode("b", "p1", 2, 1, 2, 0.0, 0.4),
            episode("c", "p1", 3, 1, 3, 0.0, 0.4),
            episode("d", "p2", 1, 1, 4, 0.0, 0.4),
        ];
        let report = evaluate_temporal_holdout(
            &records,
            TemporalHoldoutSplit {
                training_through_sequence: 2,
                testing_from_sequence: 3,
            },
            DEFAULT_PERSONAL_PRIOR_STRENGTH,
        )
        .expect("valid temporal split");

        assert_eq!(report.training_records, 2);
        assert_eq!(report.testing_records, 2);
        let valence = report
            .channels
            .iter()
            .find(|row| row.channel == ResponseChannel::Valence)
            .unwrap();
        assert_eq!(valence.scored_test_observations, 2);
        assert!((valence.population_training_mean_delta.unwrap() - 0.4).abs() < 1e-6);
        assert!((valence.no_change_mae.unwrap() - 0.4).abs() < 1e-6);
        assert_eq!(valence.personal_history_observations, 1);
        assert_eq!(valence.personal_history_coverage, 0.5);
        assert!(
            (valence.no_change_mae_on_personal_history.unwrap() - 0.4).abs() < 1e-6
        );
        assert_eq!(valence.population_delta_mae_on_personal_history, Some(0.0));
        assert_eq!(valence.personal_history_mae, Some(0.0));
    }

    #[test]
    fn future_test_outcomes_cannot_change_training_prior() {
        let train = episode("train", "p1", 1, 1, 1, 0.0, 0.2);
        let test = episode("test", "p1", 2, 1, 3, 0.0, 0.8);
        let future = episode("future", "p1", 3, 1, 4, 0.0, -0.9);
        let split = TemporalHoldoutSplit {
            training_through_sequence: 1,
            testing_from_sequence: 3,
        };
        let first = evaluate_temporal_holdout(&[train.clone(), test.clone()], split, 3.0)
            .expect("valid split");
        let with_future = evaluate_temporal_holdout(&[train, test, future], split, 3.0)
            .expect("future response remains test data");
        let first_prior = first
            .channels
            .iter()
            .find(|row| row.channel == ResponseChannel::Valence)
            .unwrap()
            .population_training_mean_delta;
        let second_prior = with_future
            .channels
            .iter()
            .find(|row| row.channel == ResponseChannel::Valence)
            .unwrap()
            .population_training_mean_delta;
        assert_eq!(first_prior, second_prior);
        assert!((first_prior.unwrap() - 0.2).abs() < 1e-6);
        assert_eq!(with_future.training_records, 1);
    }

    #[test]
    fn reversed_participant_session_order_is_rejected() {
        let later = episode("later", "p1", 2, 1, 1, 0.0, 0.3);
        let earlier = episode("earlier", "p1", 1, 1, 2, 0.0, 0.2);
        let result = evaluate_temporal_holdout(
            &[later, earlier],
            TemporalHoldoutSplit {
                training_through_sequence: 1,
                testing_from_sequence: 2,
            },
            3.0,
        );
        assert!(result.unwrap_err().iter().any(|issue| matches!(
            issue, BenchmarkIssue::ParticipantChronologyViolation { .. }
        )));
    }

    #[test]
    fn one_listener_session_cannot_cross_train_test_boundary() {
        let first = episode("part-a", "p1", 1, 1, 1, 0.0, 0.3);
        let second = episode("part-b", "p1", 1, 2, 2, 0.0, 0.2);
        let result = evaluate_temporal_holdout(
            &[first, second],
            TemporalHoldoutSplit {
                training_through_sequence: 1,
                testing_from_sequence: 2,
            },
            3.0,
        );
        assert!(result.unwrap_err().iter().any(|issue| matches!(
            issue,
            BenchmarkIssue::SessionCrossesSplitBoundary {
                participant_token,
                session_index: 1
            } if participant_token == "p1"
        )));
    }

    #[test]
    fn participant_bootstrap_counts_listeners_not_just_episodes() {
        let mut records = Vec::new();
        for index in 0..12u64 {
            let id = index + 1;
            let delta = (index as f32 + 1.0) / 100.0;
            records.push(episode(
                &format!("train-{id}"),
                &format!("p{id}"),
                1,
                1,
                id,
                0.0,
                delta,
            ));
            records.push(episode(
                &format!("test-{id}"),
                &format!("p{id}"),
                2,
                1,
                id + 12,
                0.0,
                delta,
            ));
        }
        let split = TemporalHoldoutSplit {
            training_through_sequence: 12,
            testing_from_sequence: 13,
        };
        let report = evaluate_temporal_holdout(&records, split, 3.0).unwrap();
        let valence = report
            .channels
            .iter()
            .find(|row| row.channel == ResponseChannel::Valence)
            .unwrap();
        let no_change = valence
            .participant_cluster_comparisons
            .iter()
            .find(|comparison| comparison.comparator == BaselineComparator::NoChange)
            .unwrap();
        assert_eq!(no_change.participants, 12);
        assert_eq!(no_change.matched_test_observations, 12);
        assert_eq!(no_change.participants_improved_fraction, Some(1.0));
        assert!(no_change.bootstrap_confidence_interval_95.unwrap()[0] > 0.0);
        assert_eq!(no_change.bootstrap_replicates, PARTICIPANT_BOOTSTRAP_REPLICATES);
        let population = valence
            .participant_cluster_comparisons
            .iter()
            .find(|comparison| comparison.comparator == BaselineComparator::PopulationDelta)
            .unwrap();
        assert_eq!(population.participants, 12);
        assert!(population.bootstrap_confidence_interval_95.unwrap()[0] > 0.0);
    }

    #[test]
    fn equal_participant_weight_prevents_frequent_listener_domination() {
        let grouped = BTreeMap::from([
            ("frequent".to_owned(), ParticipantErrors {
                no_change: vec![1.0; 100],
                population_delta: vec![1.0; 100],
                personal_history: vec![0.0; 100],
            }),
            ("occasional".to_owned(), ParticipantErrors {
                no_change: vec![0.1],
                population_delta: vec![0.1],
                personal_history: vec![0.0],
            }),
        ]);
        let result = participant_cluster_comparison(
            &grouped,
            BaselineComparator::NoChange,
            7,
        );
        assert_eq!(result.participants, 2);
        assert_eq!(result.matched_test_observations, 101);
        assert!((result.mean_participant_mae_improvement.unwrap() - 0.55).abs() < 1e-12);
        assert_eq!(result.bootstrap_confidence_interval_95, None);
    }

    #[test]
    fn small_participant_count_suppresses_interval() {
        let mut grouped = BTreeMap::new();
        for (index, effect) in [0.1, 0.2, 0.3].iter().enumerate() {
            grouped.insert(
                format!("p{index}"),
                ParticipantErrors {
                    no_change: vec![*effect],
                    population_delta: vec![*effect],
                    personal_history: vec![0.0],
                },
            );
        }
        let result = participant_cluster_comparison(
            &grouped,
            BaselineComparator::NoChange,
            99,
        );
        assert_eq!(result.participants, 3);
        assert_eq!(result.bootstrap_confidence_interval_95, None);
        assert_eq!(result.bootstrap_replicates, 0);
        assert!((result.mean_participant_mae_improvement.unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_is_deterministic_and_empty_clusters_have_no_interval() {
        let values: Vec<f64> = (1..=12).map(|value| value as f64 / 100.0).collect();
        let first = participant_bootstrap_ci(&values, 500, 1234);
        let second = participant_bootstrap_ci(&values, 500, 1234);
        assert_eq!(first, second);

        let empty = BTreeMap::new();
        let result = participant_cluster_comparison(
            &empty,
            BaselineComparator::NoChange,
            1234,
        );
        assert_eq!(result.participants, 0);
        assert_eq!(result.bootstrap_confidence_interval_95, None);
        assert_eq!(result.bootstrap_replicates, 0);
    }

    #[test]
    fn repeated_episode_identity_is_rejected() {
        let first = episode("same", "p1", 1, 1, 1, 0.0, 0.3);
        let second = episode("same", "p2", 1, 1, 2, 0.0, 0.2);
        let result = evaluate_temporal_holdout(
            &[first, second],
            TemporalHoldoutSplit {
                training_through_sequence: 1,
                testing_from_sequence: 2,
            },
            3.0,
        );
        assert!(result.unwrap_err().iter().any(|issue| matches!(
            issue, BenchmarkIssue::DuplicateEpisodeId { .. }
        )));
    }
}
