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

pub const LISTENER_RESPONSE_BENCHMARK_VERSION: &str = "muse-listener-response-benchmark-v1";
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelBenchmarkMetrics {
    pub channel: ResponseChannel,
    pub scored_test_observations: usize,
    pub population_training_mean_delta: Option<f64>,
    pub no_change_mae: Option<f64>,
    pub population_delta_mae: Option<f64>,
    /// Only test episodes with at least one earlier session from this listener.
    pub personal_history_mae: Option<f64>,
    /// Population mean shrunk toward personal history; no-change is the fallback
    /// only when no training target exists for this channel.
    pub shrunk_personal_mae: Option<f64>,
    pub personal_history_observations: usize,
    pub personal_history_coverage: f64,
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
                    participant_token: participant.clone(),
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
        let mut shrinkage_errors = Vec::new();
        let mut personal_history_observations = 0usize;
        let mut scored_test_observations = 0usize;

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
                    personal_errors.push((actual_delta - personal_mean).abs());
                    let weight =
                        history.len() as f64 / (history.len() as f64 + prior_strength);
                    prior + weight * (personal_mean - prior)
                }
                None => prior,
            };
            shrinkage_errors.push((actual_delta - prediction).abs());
        }

        channels.push(ChannelBenchmarkMetrics {
            channel,
            scored_test_observations,
            population_training_mean_delta: population_mean,
            no_change_mae: mean(&no_change_errors),
            population_delta_mae: mean(&population_errors),
            personal_history_mae: mean(&personal_errors),
            shrunk_personal_mae: mean(&shrinkage_errors),
            personal_history_observations,
            personal_history_coverage: if scored_test_observations == 0 {
                0.0
            } else {
                personal_history_observations as f64 / scored_test_observations as f64
            },
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
