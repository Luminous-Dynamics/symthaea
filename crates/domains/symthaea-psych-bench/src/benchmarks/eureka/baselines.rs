// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent shortcut baselines for EUREKA-002B.
//!
//! These baselines consume only public transition records from development or
//! calibration partitions. They deliberately do not call Symthaea HDC, CfC,
//! causal reasoning, hidden role maps, or evaluator oracle state.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2094>

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::action_execution::{ActionExecutionStatus, QualifiedActionReceipt};
use super::consequence::{ConsequencePrediction, PredictionOutcome};
use super::hidden_world::{
    CorpusPartition, EvaluatorWorld, FixtureFamily, PublicAction, PublicObservation, PublicValue,
    RuntimeWorld, WorldBuildProfile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ShortcutBaselineKind {
    ActionMarginalDelta,
    ExactLookup,
    NearestTransition,
    SimpleMarkov,
}

impl ShortcutBaselineKind {
    pub(super) const ALL: [Self; 4] = [
        Self::ActionMarginalDelta,
        Self::ExactLookup,
        Self::NearestTransition,
        Self::SimpleMarkov,
    ];

    pub(super) const fn stable_id(self) -> &'static str {
        match self {
            Self::ActionMarginalDelta => "EUREKA.BASELINE.ACTION_MARGINAL_DELTA.v1",
            Self::ExactLookup => "EUREKA.BASELINE.EXACT_LOOKUP.v1",
            Self::NearestTransition => "EUREKA.BASELINE.NEAREST_TRANSITION.v1",
            Self::SimpleMarkov => "EUREKA.BASELINE.SIMPLE_MARKOV.v1",
        }
    }
}

/// Owns both the frozen evaluator build profile and its corresponding world.
/// Campaign code cannot supply partition/family labels when recording a
/// transition; those labels are copied from the profile that built the world.
pub(super) struct TransitionRecorder {
    profile: WorldBuildProfile,
    evaluator: EvaluatorWorld,
}

impl TransitionRecorder {
    pub(super) fn build(profile: WorldBuildProfile) -> Self {
        Self {
            profile,
            evaluator: EvaluatorWorld::build(profile),
        }
    }

    pub(super) fn world_digest(&self) -> u64 {
        self.evaluator.world_digest()
    }

    pub(super) fn runtime(&mut self) -> RuntimeWorld<'_> {
        self.evaluator.runtime()
    }

    /// Execute through the canonical realized-action route and bind the result
    /// to the exact family/partition that built this evaluator.
    pub(super) fn execute_and_record(
        &mut self,
        action: PublicAction,
    ) -> Result<(QualifiedActionReceipt, PublicTransitionRecord), TransitionRecordError> {
        let receipt = self.evaluator.execute_qualified_action(action);
        let record = PublicTransitionRecord::from_bound_receipt(self.profile, &receipt)?;
        Ok((receipt, record))
    }
}

/// Evaluator-side public transition record. It contains no hidden state or
/// mechanism labels. Metadata is evaluator-bound and fields are opaque outside
/// this module to prevent accidental relabeling after capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PublicTransitionRecord {
    world_digest: u64,
    transition_digest: u64,
    partition: CorpusPartition,
    family: FixtureFamily,
    pre_state: PublicObservation,
    action: PublicAction,
    post_state: PublicObservation,
}

impl PublicTransitionRecord {
    fn from_bound_receipt(
        profile: WorldBuildProfile,
        receipt: &QualifiedActionReceipt,
    ) -> Result<Self, TransitionRecordError> {
        if receipt.status != ActionExecutionStatus::Applied {
            return Err(TransitionRecordError::ActionNotApplied);
        }
        let realized = receipt
            .realized
            .ok_or(TransitionRecordError::MissingRealizedAction)?;
        if realized != receipt.requested {
            return Err(TransitionRecordError::RequestRealizationMismatch);
        }
        let post_state = receipt
            .post_state
            .clone()
            .ok_or(TransitionRecordError::MissingPostState)?;
        let transition_digest = receipt
            .transition_digest
            .ok_or(TransitionRecordError::MissingTransitionIdentity)?;
        Ok(Self {
            world_digest: receipt.world_digest,
            transition_digest,
            partition: profile.partition,
            family: profile.family,
            pre_state: receipt.pre_state.clone(),
            action: realized,
            post_state,
        })
    }

    pub(super) fn world_digest(&self) -> u64 {
        self.world_digest
    }

    pub(super) fn transition_digest(&self) -> u64 {
        self.transition_digest
    }

    pub(super) fn partition(&self) -> CorpusPartition {
        self.partition
    }

    pub(super) fn family(&self) -> FixtureFamily {
        self.family
    }

    pub(super) fn pre_state(&self) -> &PublicObservation {
        &self.pre_state
    }

    pub(super) fn action(&self) -> PublicAction {
        self.action
    }

    pub(super) fn post_state(&self) -> &PublicObservation {
        &self.post_state
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TransitionRecordError {
    ActionNotApplied,
    MissingRealizedAction,
    MissingPostState,
    MissingTransitionIdentity,
    RequestRealizationMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CorpusError {
    Empty,
    DisallowedFitPartition,
    DisallowedHeldOutPartition,
    DuplicateTransition,
    FieldCountMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BaselineFitCorpus {
    records: Vec<PublicTransitionRecord>,
    digest: u64,
}

impl BaselineFitCorpus {
    pub(super) fn freeze(mut records: Vec<PublicTransitionRecord>) -> Result<Self, CorpusError> {
        validate_records(&records)?;
        if records.iter().any(|record| {
            !matches!(
                record.partition,
                CorpusPartition::Development | CorpusPartition::Calibration
            )
        }) {
            return Err(CorpusError::DisallowedFitPartition);
        }
        records.sort_by_key(|record| record.transition_digest);
        let digest = corpus_digest(b"eureka.baseline-fit-corpus.v1\0", &records);
        Ok(Self { records, digest })
    }

    pub(super) fn digest(&self) -> u64 {
        self.digest
    }

    pub(super) fn len(&self) -> usize {
        self.records.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HeldOutTransitionCorpus {
    records: Vec<PublicTransitionRecord>,
    digest: u64,
}

impl HeldOutTransitionCorpus {
    pub(super) fn freeze(mut records: Vec<PublicTransitionRecord>) -> Result<Self, CorpusError> {
        validate_records(&records)?;
        if records.iter().any(|record| {
            !matches!(
                record.partition,
                CorpusPartition::HeldOutEvaluation | CorpusPartition::ExternalReplication
            )
        }) {
            return Err(CorpusError::DisallowedHeldOutPartition);
        }
        records.sort_by_key(|record| record.transition_digest);
        let digest = corpus_digest(b"eureka.held-out-transition-corpus.v1\0", &records);
        Ok(Self { records, digest })
    }

    pub(super) fn digest(&self) -> u64 {
        self.digest
    }

    pub(super) fn records(&self) -> &[PublicTransitionRecord] {
        &self.records
    }
}

fn validate_records(records: &[PublicTransitionRecord]) -> Result<(), CorpusError> {
    if records.is_empty() {
        return Err(CorpusError::Empty);
    }
    let mut seen = HashSet::new();
    for record in records {
        if record.pre_state.fields.len() != record.post_state.fields.len() {
            return Err(CorpusError::FieldCountMismatch);
        }
        if !seen.insert(record.transition_digest) {
            return Err(CorpusError::DuplicateTransition);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FittedShortcutBaselines {
    fit_corpus_digest: u64,
    records: Vec<PublicTransitionRecord>,
}

impl FittedShortcutBaselines {
    pub(super) fn fit(corpus: &BaselineFitCorpus) -> Self {
        Self {
            fit_corpus_digest: corpus.digest,
            records: corpus.records.clone(),
        }
    }

    pub(super) fn fit_corpus_digest(&self) -> u64 {
        self.fit_corpus_digest
    }

    pub(super) fn predict(
        &self,
        kind: ShortcutBaselineKind,
        pre: &PublicObservation,
        action: PublicAction,
    ) -> ConsequencePrediction {
        let fields = match kind {
            ShortcutBaselineKind::ActionMarginalDelta => self.action_marginal_delta(pre, action),
            ShortcutBaselineKind::ExactLookup => self.exact_lookup(pre, action),
            ShortcutBaselineKind::NearestTransition => self.nearest_transition(pre, action),
            ShortcutBaselineKind::SimpleMarkov => self.simple_markov(pre, action),
        };
        ConsequencePrediction {
            action,
            outcome: match fields {
                Some(fields) => PredictionOutcome::Predicted { fields },
                None => PredictionOutcome::AbstainInsufficientEvidence,
            },
        }
    }

    fn exact_lookup(
        &self,
        pre: &PublicObservation,
        action: PublicAction,
    ) -> Option<Vec<PublicValue>> {
        let matches: Vec<&PublicTransitionRecord> = self
            .records
            .iter()
            .filter(|record| record.action == action && record.pre_state == *pre)
            .collect();
        choose_modal_post(&matches)
    }

    fn nearest_transition(
        &self,
        pre: &PublicObservation,
        action: PublicAction,
    ) -> Option<Vec<PublicValue>> {
        self.records
            .iter()
            .filter(|record| record.action == action)
            .filter_map(|record| {
                public_distance(&record.pre_state, pre)
                    .map(|distance| (distance, record.transition_digest, record))
            })
            .min_by_key(|(distance, digest, _)| (*distance, *digest))
            .map(|(_, _, record)| record.post_state.fields.clone())
    }

    fn action_marginal_delta(
        &self,
        pre: &PublicObservation,
        action: PublicAction,
    ) -> Option<Vec<PublicValue>> {
        let candidates: Vec<&PublicTransitionRecord> = self
            .records
            .iter()
            .filter(|record| {
                record.action == action
                    && record.pre_state.fields.len() == pre.fields.len()
                    && record.post_state.fields.len() == pre.fields.len()
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }

        let mut output = Vec::with_capacity(pre.fields.len());
        for index in 0..pre.fields.len() {
            let mut counts: HashMap<FieldTransform, usize> = HashMap::new();
            for record in &candidates {
                let transform = FieldTransform::between(
                    record.pre_state.fields[index],
                    record.post_state.fields[index],
                );
                *counts.entry(transform).or_insert(0) += 1;
            }
            let transform = choose_mode_transform(&counts)?;
            output.push(transform.apply(pre.fields[index]));
        }
        Some(output)
    }

    fn simple_markov(
        &self,
        pre: &PublicObservation,
        action: PublicAction,
    ) -> Option<Vec<PublicValue>> {
        let mut output = Vec::with_capacity(pre.fields.len());
        for index in 0..pre.fields.len() {
            let mut counts: HashMap<PublicValue, usize> = HashMap::new();
            for record in &self.records {
                if record.action != action
                    || record.pre_state.fields.len() != pre.fields.len()
                    || record.post_state.fields.len() != pre.fields.len()
                    || record.pre_state.fields[index] != pre.fields[index]
                {
                    continue;
                }
                *counts.entry(record.post_state.fields[index]).or_insert(0) += 1;
            }
            output.push(choose_mode_value(&counts)?);
        }
        Some(output)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FieldTransform {
    Same,
    FlipBit,
    CountDelta(i32),
    Replace(PublicValue),
}

impl FieldTransform {
    fn between(before: PublicValue, after: PublicValue) -> Self {
        match (before, after) {
            (PublicValue::Bit(a), PublicValue::Bit(b)) => {
                if a == b { Self::Same } else { Self::FlipBit }
            }
            (PublicValue::Count(a), PublicValue::Count(b)) => b
                .checked_sub(a)
                .map(Self::CountDelta)
                .unwrap_or(Self::Replace(after)),
            (_, _) if before == after => Self::Same,
            (_, _) => Self::Replace(after),
        }
    }

    fn apply(self, before: PublicValue) -> PublicValue {
        match (self, before) {
            (Self::Same, value) => value,
            (Self::FlipBit, PublicValue::Bit(value)) => PublicValue::Bit(!value),
            (Self::CountDelta(delta), PublicValue::Count(value)) => value
                .checked_add(delta)
                .map(PublicValue::Count)
                .unwrap_or(PublicValue::Count(value)),
            (Self::Replace(value), _) => value,
            (_, value) => value,
        }
    }
}

fn choose_modal_post(records: &[&PublicTransitionRecord]) -> Option<Vec<PublicValue>> {
    if records.is_empty() {
        return None;
    }
    let mut counts: HashMap<Vec<PublicValue>, usize> = HashMap::new();
    for record in records {
        *counts.entry(record.post_state.fields.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|(a_values, a_count), (b_values, b_count)| {
            a_count
                .cmp(b_count)
                .then_with(|| compare_value_slices(b_values, a_values))
        })
        .map(|(values, _)| values)
}

fn choose_mode_value(counts: &HashMap<PublicValue, usize>) -> Option<PublicValue> {
    counts
        .iter()
        .max_by(|(a_value, a_count), (b_value, b_count)| {
            a_count
                .cmp(b_count)
                .then_with(|| compare_value(**b_value, **a_value))
        })
        .map(|(value, _)| *value)
}

fn choose_mode_transform(counts: &HashMap<FieldTransform, usize>) -> Option<FieldTransform> {
    counts
        .iter()
        .max_by(|(a_value, a_count), (b_value, b_count)| {
            a_count
                .cmp(b_count)
                .then_with(|| transform_key(**b_value).cmp(&transform_key(**a_value)))
        })
        .map(|(value, _)| *value)
}

fn public_distance(a: &PublicObservation, b: &PublicObservation) -> Option<u64> {
    if a.fields.len() != b.fields.len() {
        return None;
    }
    let mut total = 0_u64;
    for (left, right) in a.fields.iter().zip(&b.fields) {
        let distance = match (*left, *right) {
            (PublicValue::Bit(x), PublicValue::Bit(y)) => u64::from(x != y),
            (PublicValue::Count(x), PublicValue::Count(y)) => {
                i64::from(x).abs_diff(i64::from(y))
            }
            _ => return None,
        };
        total = total.saturating_add(distance);
    }
    Some(total)
}

fn compare_value_slices(a: &[PublicValue], b: &[PublicValue]) -> Ordering {
    for (left, right) in a.iter().zip(b) {
        let order = compare_value(*left, *right);
        if order != Ordering::Equal {
            return order;
        }
    }
    a.len().cmp(&b.len())
}

fn compare_value(a: PublicValue, b: PublicValue) -> Ordering {
    value_key(a).cmp(&value_key(b))
}

fn value_key(value: PublicValue) -> (u8, i64) {
    match value {
        PublicValue::Bit(false) => (0, 0),
        PublicValue::Bit(true) => (0, 1),
        PublicValue::Count(value) => (1, i64::from(value)),
    }
}

fn transform_key(value: FieldTransform) -> (u8, i64, u8, i64) {
    match value {
        FieldTransform::Same => (0, 0, 0, 0),
        FieldTransform::FlipBit => (1, 0, 0, 0),
        FieldTransform::CountDelta(delta) => (2, i64::from(delta), 0, 0),
        FieldTransform::Replace(value) => {
            let (tag, scalar) = value_key(value);
            (3, 0, tag, scalar)
        }
    }
}

fn corpus_digest(domain: &[u8], records: &[PublicTransitionRecord]) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for record in records {
        bytes.extend_from_slice(&record.transition_digest.to_le_bytes());
        bytes.push(match record.partition {
            CorpusPartition::Development => 1,
            CorpusPartition::Calibration => 2,
            CorpusPartition::HeldOutEvaluation => 3,
            CorpusPartition::ExternalReplication => 4,
        });
    }
    fnv1a64(&bytes)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(step: u32, fields: Vec<PublicValue>) -> PublicObservation {
        PublicObservation { step, fields }
    }

    fn record(
        digest: u64,
        partition: CorpusPartition,
        pre: Vec<PublicValue>,
        action: PublicAction,
        post: Vec<PublicValue>,
    ) -> PublicTransitionRecord {
        PublicTransitionRecord {
            world_digest: 10 + digest,
            transition_digest: digest,
            partition,
            family: FixtureFamily::CausalBits,
            pre_state: obs(0, pre),
            action,
            post_state: obs(1, post),
        }
    }

    fn predicted_fields(prediction: ConsequencePrediction) -> Option<Vec<PublicValue>> {
        match prediction.outcome {
            PredictionOutcome::Predicted { fields } => Some(fields),
            _ => None,
        }
    }

    #[test]
    fn recorder_binds_partition_and_family_from_build_profile() {
        let profile = WorldBuildProfile {
            family: FixtureFamily::ResourceFlow,
            seed: 8,
            mechanism_variant: 0,
            partition: CorpusPartition::ExternalReplication,
        };
        let mut recorder = TransitionRecorder::build(profile);
        let (_, record) = recorder.execute_and_record(PublicAction::NoOp).unwrap();
        assert_eq!(record.partition(), CorpusPartition::ExternalReplication);
        assert_eq!(record.family(), FixtureFamily::ResourceFlow);
        assert_eq!(record.world_digest(), recorder.world_digest());
    }

    #[test]
    fn fit_rejects_held_out_partitions() {
        let held_out = record(
            1,
            CorpusPartition::HeldOutEvaluation,
            vec![PublicValue::Bit(false)],
            PublicAction::NoOp,
            vec![PublicValue::Bit(true)],
        );
        assert_eq!(
            BaselineFitCorpus::freeze(vec![held_out]),
            Err(CorpusError::DisallowedFitPartition)
        );
    }

    #[test]
    fn held_out_corpus_rejects_development_records() {
        let development = record(
            1,
            CorpusPartition::Development,
            vec![PublicValue::Bit(false)],
            PublicAction::NoOp,
            vec![PublicValue::Bit(true)],
        );
        assert_eq!(
            HeldOutTransitionCorpus::freeze(vec![development]),
            Err(CorpusError::DisallowedHeldOutPartition)
        );
    }

    #[test]
    fn duplicate_transition_identity_is_rejected() {
        let a = record(
            7,
            CorpusPartition::Development,
            vec![PublicValue::Bit(false)],
            PublicAction::NoOp,
            vec![PublicValue::Bit(true)],
        );
        let mut b = a.clone();
        b.world_digest = 999;
        assert_eq!(
            BaselineFitCorpus::freeze(vec![a, b]),
            Err(CorpusError::DuplicateTransition)
        );
    }

    #[test]
    fn fit_corpus_identity_is_order_invariant() {
        let a = record(
            2,
            CorpusPartition::Development,
            vec![PublicValue::Bit(false)],
            PublicAction::NoOp,
            vec![PublicValue::Bit(true)],
        );
        let b = record(
            1,
            CorpusPartition::Calibration,
            vec![PublicValue::Bit(true)],
            PublicAction::NoOp,
            vec![PublicValue::Bit(false)],
        );
        let ab = BaselineFitCorpus::freeze(vec![a.clone(), b.clone()]).unwrap();
        let ba = BaselineFitCorpus::freeze(vec![b, a]).unwrap();
        assert_eq!(ab.digest(), ba.digest());
    }

    #[test]
    fn exact_lookup_reaches_replay_ceiling_and_abstains_unseen() {
        let seen = record(
            1,
            CorpusPartition::Development,
            vec![PublicValue::Count(2)],
            PublicAction::NoOp,
            vec![PublicValue::Count(3)],
        );
        let corpus = BaselineFitCorpus::freeze(vec![seen.clone()]).unwrap();
        let fit = FittedShortcutBaselines::fit(&corpus);
        assert_eq!(
            predicted_fields(fit.predict(
                ShortcutBaselineKind::ExactLookup,
                seen.pre_state(),
                PublicAction::NoOp,
            )),
            Some(seen.post_state().fields.clone())
        );
        let unseen = obs(0, vec![PublicValue::Count(9)]);
        assert_eq!(
            fit.predict(
                ShortcutBaselineKind::ExactLookup,
                &unseen,
                PublicAction::NoOp,
            )
            .outcome,
            PredictionOutcome::AbstainInsufficientEvidence
        );
    }

    #[test]
    fn nearest_transition_ties_break_by_transition_identity() {
        let low = record(
            10,
            CorpusPartition::Development,
            vec![PublicValue::Count(0)],
            PublicAction::NoOp,
            vec![PublicValue::Count(10)],
        );
        let high = record(
            20,
            CorpusPartition::Development,
            vec![PublicValue::Count(2)],
            PublicAction::NoOp,
            vec![PublicValue::Count(20)],
        );
        let corpus = BaselineFitCorpus::freeze(vec![high, low]).unwrap();
        let fit = FittedShortcutBaselines::fit(&corpus);
        let query = obs(99, vec![PublicValue::Count(1)]);
        assert_eq!(
            predicted_fields(fit.predict(
                ShortcutBaselineKind::NearestTransition,
                &query,
                PublicAction::NoOp,
            )),
            Some(vec![PublicValue::Count(10)])
        );
    }

    #[test]
    fn action_marginal_delta_applies_learned_change_to_new_state() {
        let a = record(
            1,
            CorpusPartition::Development,
            vec![PublicValue::Count(2)],
            PublicAction::NoOp,
            vec![PublicValue::Count(3)],
        );
        let b = record(
            2,
            CorpusPartition::Calibration,
            vec![PublicValue::Count(7)],
            PublicAction::NoOp,
            vec![PublicValue::Count(8)],
        );
        let fit = FittedShortcutBaselines::fit(&BaselineFitCorpus::freeze(vec![a, b]).unwrap());
        let query = obs(0, vec![PublicValue::Count(100)]);
        assert_eq!(
            predicted_fields(fit.predict(
                ShortcutBaselineKind::ActionMarginalDelta,
                &query,
                PublicAction::NoOp,
            )),
            Some(vec![PublicValue::Count(101)])
        );
    }

    #[test]
    fn simple_markov_conditions_on_current_public_value() {
        let a = record(
            1,
            CorpusPartition::Development,
            vec![PublicValue::Bit(false)],
            PublicAction::NoOp,
            vec![PublicValue::Bit(true)],
        );
        let b = record(
            2,
            CorpusPartition::Calibration,
            vec![PublicValue::Bit(true)],
            PublicAction::NoOp,
            vec![PublicValue::Bit(false)],
        );
        let fit = FittedShortcutBaselines::fit(&BaselineFitCorpus::freeze(vec![a, b]).unwrap());
        assert_eq!(
            predicted_fields(fit.predict(
                ShortcutBaselineKind::SimpleMarkov,
                &obs(7, vec![PublicValue::Bit(false)]),
                PublicAction::NoOp,
            )),
            Some(vec![PublicValue::Bit(true)])
        );
    }

    #[test]
    fn held_out_outcome_cannot_change_a_frozen_fit_prediction() {
        let training = record(
            1,
            CorpusPartition::Development,
            vec![PublicValue::Count(0)],
            PublicAction::NoOp,
            vec![PublicValue::Count(1)],
        );
        let fit = FittedShortcutBaselines::fit(&BaselineFitCorpus::freeze(vec![training]).unwrap());
        let query = obs(0, vec![PublicValue::Count(0)]);
        let before = fit.predict(ShortcutBaselineKind::ExactLookup, &query, PublicAction::NoOp);

        let mut held_out = record(
            2,
            CorpusPartition::HeldOutEvaluation,
            vec![PublicValue::Count(0)],
            PublicAction::NoOp,
            vec![PublicValue::Count(999)],
        );
        held_out.post_state.fields[0] = PublicValue::Count(-999);
        let after = fit.predict(ShortcutBaselineKind::ExactLookup, &query, PublicAction::NoOp);
        assert_eq!(before, after);
    }
}
