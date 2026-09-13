// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen Development-only shortcut-comparator subject for EUREKA-002 V2.
//!
//! This module consumes only canonical Development evidence. Once frozen, the
//! subject exposes prediction only; it has no Calibration/target/fit mutation
//! API and no dependency on Symthaea/FEP internals.

use std::collections::BTreeMap;

use super::baselines::ShortcutBaselineKind;
use super::consequence::{ConsequencePrediction, PredictionOutcome};
use super::hidden_world::{PublicAction, PublicValue};
use super::v2_comparator_custody::{
    V2DevelopmentFitCorpus, V2PublicTransitionEvidence,
    V2_SHORTCUT_BASELINE_IMPLEMENTATION_REVISION,
};
use super::v2_public_schema::{V2_OBSERVATION_DIM, V2PublicFamily, V2PublicState};

pub(super) const V2_FROZEN_COMPARATOR_SUBJECT_REVISION: &str =
    "EUREKA.002.V2.FROZEN_COMPARATOR_SUBJECT.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrozenFitRow {
    family: V2PublicFamily,
    row_identity: [u8; 32],
    pre: V2PublicState,
    action: PublicAction,
    post: V2PublicState,
}

impl From<&V2PublicTransitionEvidence> for FrozenFitRow {
    fn from(value: &V2PublicTransitionEvidence) -> Self {
        Self {
            family: value.family(),
            row_identity: value.row_identity(),
            pre: value.pre(),
            action: value.action(),
            post: value.post(),
        }
    }
}

/// Immutable prediction-only shortcut-comparator subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2FrozenComparatorSubject {
    schema_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    records: Vec<FrozenFitRow>,
    commitment: [u8; 32],
}

impl V2FrozenComparatorSubject {
    pub(super) fn freeze(corpus: &V2DevelopmentFitCorpus) -> Self {
        let records: Vec<_> = corpus.records().iter().map(FrozenFitRow::from).collect();
        let commitment = subject_commitment(
            corpus.schema_commitment(),
            corpus.commitment(),
            records.len(),
        );
        Self {
            schema_commitment: corpus.schema_commitment(),
            fit_corpus_commitment: corpus.commitment(),
            records,
            commitment,
        }
    }

    pub(super) const fn schema_commitment(&self) -> [u8; 32] {
        self.schema_commitment
    }

    pub(super) const fn fit_corpus_commitment(&self) -> [u8; 32] {
        self.fit_corpus_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn predict(
        &self,
        kind: ShortcutBaselineKind,
        family: V2PublicFamily,
        pre: V2PublicState,
        action: PublicAction,
    ) -> ConsequencePrediction {
        let predicted = match kind {
            ShortcutBaselineKind::ActionMarginalDelta => {
                self.action_marginal_delta(family, pre, action)
            }
            ShortcutBaselineKind::ExactLookup => self.exact_lookup(family, pre, action),
            ShortcutBaselineKind::NearestTransition => {
                self.nearest_transition(family, pre, action)
            }
            ShortcutBaselineKind::SimpleMarkov => self.simple_markov(family, pre, action),
        };

        ConsequencePrediction {
            action,
            outcome: match predicted {
                Some(fields) => PredictionOutcome::Predicted {
                    fields: fields.into_iter().map(PublicValue::Count).collect(),
                },
                None => PredictionOutcome::AbstainInsufficientEvidence,
            },
        }
    }

    fn exact_lookup(
        &self,
        family: V2PublicFamily,
        pre: V2PublicState,
        action: PublicAction,
    ) -> Option<[i32; V2_OBSERVATION_DIM]> {
        let mut counts = BTreeMap::<[i32; V2_OBSERVATION_DIM], u32>::new();
        for record in self.records.iter().filter(|record| {
            record.family == family && record.action == action && record.pre == pre
        }) {
            *counts.entry(record.post.fields()).or_default() += 1;
        }
        choose_mode(&counts)
    }

    fn nearest_transition(
        &self,
        family: V2PublicFamily,
        pre: V2PublicState,
        action: PublicAction,
    ) -> Option<[i32; V2_OBSERVATION_DIM]> {
        self.records
            .iter()
            .filter(|record| record.family == family && record.action == action)
            .map(|record| {
                (
                    distance(pre, record.pre),
                    record.row_identity,
                    record.post.fields(),
                )
            })
            .min_by_key(|(distance, row_identity, _)| (*distance, *row_identity))
            .map(|(_, _, post)| post)
    }

    fn action_marginal_delta(
        &self,
        family: V2PublicFamily,
        pre: V2PublicState,
        action: PublicAction,
    ) -> Option<[i32; V2_OBSERVATION_DIM]> {
        let candidates: Vec<_> = self
            .records
            .iter()
            .filter(|record| record.family == family && record.action == action)
            .collect();
        if candidates.is_empty() {
            return None;
        }

        let mut output = pre.fields();
        for (field, slot) in output.iter_mut().enumerate() {
            let mut counts = BTreeMap::<i32, u32>::new();
            for record in &candidates {
                let delta = record.post.fields()[field]
                    .saturating_sub(record.pre.fields()[field]);
                *counts.entry(delta).or_default() += 1;
            }
            let delta = choose_mode(&counts)?;
            *slot = slot.saturating_add(delta);
        }
        Some(output)
    }

    fn simple_markov(
        &self,
        family: V2PublicFamily,
        pre: V2PublicState,
        action: PublicAction,
    ) -> Option<[i32; V2_OBSERVATION_DIM]> {
        let pre_fields = pre.fields();
        let mut output = pre_fields;
        for (field, slot) in output.iter_mut().enumerate() {
            let mut counts = BTreeMap::<i32, u32>::new();
            for record in self.records.iter().filter(|record| {
                record.family == family
                    && record.action == action
                    && record.pre.fields()[field] == pre_fields[field]
            }) {
                *counts.entry(record.post.fields()[field]).or_default() += 1;
            }
            *slot = choose_mode(&counts)?;
        }
        Some(output)
    }
}

fn choose_mode<T: Copy + Ord>(counts: &BTreeMap<T, u32>) -> Option<T> {
    counts
        .iter()
        .max_by(|(left_value, left_count), (right_value, right_count)| {
            left_count
                .cmp(right_count)
                // Equal-frequency ties choose the lexicographically/numerically
                // smaller value, matching the frozen construct comparator rule.
                .then_with(|| right_value.cmp(left_value))
        })
        .map(|(value, _)| *value)
}

fn distance(left: V2PublicState, right: V2PublicState) -> u64 {
    left.fields()
        .into_iter()
        .zip(right.fields())
        .map(|(a, b)| i64::from(a).abs_diff(i64::from(b)))
        .sum()
}

fn subject_commitment(
    schema_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    record_count: usize,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_FROZEN_COMPARATOR_SUBJECT_REVISION.as_bytes());
    encode_bytes(
        &mut bytes,
        V2_SHORTCUT_BASELINE_IMPLEMENTATION_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&schema_commitment);
    bytes.extend_from_slice(&fit_corpus_commitment);
    bytes.extend_from_slice(&(record_count as u64).to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::v2_comparator_custody::{V2CorpusPartition, V2PublicTransitionEvidence};

    fn record(
        family: V2PublicFamily,
        pre: [i32; 4],
        action: PublicAction,
        post: [i32; 4],
    ) -> V2PublicTransitionEvidence {
        V2PublicTransitionEvidence::new(
            family,
            V2CorpusPartition::Development,
            V2PublicState::new(pre).unwrap(),
            action,
            V2PublicState::new(post).unwrap(),
        )
        .unwrap()
    }

    fn corpus() -> V2DevelopmentFitCorpus {
        V2DevelopmentFitCorpus::freeze(vec![
            record(
                V2PublicFamily::PublicFlowV2,
                [1, 2, 3, 0],
                PublicAction::Pulse { slot: 0 },
                [0, 3, 3, 0],
            ),
            record(
                V2PublicFamily::PublicFlowV2,
                [5, 2, 3, 0],
                PublicAction::Pulse { slot: 0 },
                [4, 3, 3, 0],
            ),
            record(
                V2PublicFamily::PublicRelayV2,
                [1, 9, 4, 5],
                PublicAction::Pulse { slot: 0 },
                [2, 2, 4, 5],
            ),
        ])
        .unwrap()
    }

    #[test]
    fn subject_identity_is_deterministic_and_binds_exact_fit_corpus() {
        let corpus = corpus();
        let a = V2FrozenComparatorSubject::freeze(&corpus);
        let b = V2FrozenComparatorSubject::freeze(&corpus);
        assert_eq!(a, b);
        assert_eq!(a.schema_commitment(), corpus.schema_commitment());
        assert_eq!(a.fit_corpus_commitment(), corpus.commitment());
        assert_ne!(a.commitment(), [0_u8; 32]);

        let changed = V2DevelopmentFitCorpus::freeze(vec![record(
            V2PublicFamily::PublicFlowV2,
            [1, 2, 3, 0],
            PublicAction::Pulse { slot: 0 },
            [1, 3, 3, 0],
        )])
        .unwrap();
        assert_ne!(
            a.commitment(),
            V2FrozenComparatorSubject::freeze(&changed).commitment()
        );
    }

    #[test]
    fn exact_lookup_abstains_on_unseen_state_instead_of_copying() {
        let subject = V2FrozenComparatorSubject::freeze(&corpus());
        let prediction = subject.predict(
            ShortcutBaselineKind::ExactLookup,
            V2PublicFamily::PublicFlowV2,
            V2PublicState::new([7, 2, 3, 0]).unwrap(),
            PublicAction::Pulse { slot: 0 },
        );
        assert_eq!(prediction.outcome, PredictionOutcome::AbstainInsufficientEvidence);
    }

    #[test]
    fn action_marginal_delta_is_family_scoped() {
        let subject = V2FrozenComparatorSubject::freeze(&corpus());
        let prediction = subject.predict(
            ShortcutBaselineKind::ActionMarginalDelta,
            V2PublicFamily::PublicFlowV2,
            V2PublicState::new([9, 2, 3, 0]).unwrap(),
            PublicAction::Pulse { slot: 0 },
        );
        assert_eq!(
            prediction.outcome,
            PredictionOutcome::Predicted {
                fields: vec![
                    PublicValue::Count(8),
                    PublicValue::Count(3),
                    PublicValue::Count(3),
                    PublicValue::Count(0),
                ]
            }
        );
    }

    #[test]
    fn nearest_transition_tie_breaks_by_canonical_row_identity() {
        let fit = V2DevelopmentFitCorpus::freeze(vec![
            record(
                V2PublicFamily::PublicFlowV2,
                [1, 0, 0, 0],
                PublicAction::NoOp,
                [1, 1, 0, 0],
            ),
            record(
                V2PublicFamily::PublicFlowV2,
                [3, 0, 0, 0],
                PublicAction::NoOp,
                [3, 0, 1, 0],
            ),
        ])
        .unwrap();
        let subject = V2FrozenComparatorSubject::freeze(&fit);
        let query = V2PublicState::new([2, 0, 0, 0]).unwrap();
        let expected = fit
            .records()
            .iter()
            .min_by_key(|record| record.row_identity())
            .unwrap()
            .post()
            .fields();
        let prediction = subject.predict(
            ShortcutBaselineKind::NearestTransition,
            V2PublicFamily::PublicFlowV2,
            query,
            PublicAction::NoOp,
        );
        assert_eq!(
            prediction.outcome,
            PredictionOutcome::Predicted {
                fields: expected.into_iter().map(PublicValue::Count).collect(),
            }
        );
    }

    #[test]
    fn simple_markov_abstains_if_any_field_has_no_matching_public_evidence() {
        let subject = V2FrozenComparatorSubject::freeze(&corpus());
        let prediction = subject.predict(
            ShortcutBaselineKind::SimpleMarkov,
            V2PublicFamily::PublicFlowV2,
            V2PublicState::new([99, 2, 3, 0]).err().map_or_else(
                || unreachable!("99 is outside public schema"),
                |_| V2PublicState::new([7, 7, 7, 0]).unwrap(),
            ),
            PublicAction::Pulse { slot: 0 },
        );
        assert_eq!(prediction.outcome, PredictionOutcome::AbstainInsufficientEvidence);
    }
}
