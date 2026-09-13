// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Target-blind RelayTriad comparator fit + Calibration selection.
//!
//! Development public transitions fit four shortcut controls. Fresh evaluator-
//! only Calibration transitions choose the primary comparator. No production-
//! FEP target prediction is constructed on the Calibration path.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2245>

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::baselines::ShortcutBaselineKind;
use super::consequence::{
    ConsequencePrediction, ConsequenceScore, ConsequenceScoringError, PredictionOutcome,
    score_consequence,
};
use super::hidden_world::{CorpusPartition, PublicAction, PublicObservation, PublicValue};
use super::relay_triad::{
    RELAY_CALIBRATION_WORLDS, RELAY_TRIAD_FAMILY_ID, RELAY_TRIAD_SCHEDULE_REVISION,
    RelayTriadEvaluator, RelayTriadProfile, relay_scheduled_action, relay_scheduled_profiles,
};
use super::relay_triad_development::{
    RelayDevelopmentArtifact, RelayDevelopmentTransitionRecord,
};
use super::selection::{
    ComparatorCalibrationReceipt, PrimaryComparatorSelection, PrimaryComparatorSelectionStatus,
};

pub(super) const RELAY_COMPARATOR_FREEZE_REVISION: &str =
    "EUREKA.002O.RELAY_TRIAD_COMPARATOR_FREEZE.v1";
pub(super) const RELAY_COMPARATOR_SELECTION_REVISION: &str =
    "EUREKA.002O.RELAY_TRIAD_CALIBRATION_SELECTION.v1";
pub(super) const RELAY_TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE: u32 = 0;

#[derive(Debug, Clone, PartialEq, Eq)]
struct RelayBaselineRecord {
    transition_digest: u64,
    partition: CorpusPartition,
    pre_state: PublicObservation,
    action: PublicAction,
    post_state: PublicObservation,
}

impl From<&RelayDevelopmentTransitionRecord> for RelayBaselineRecord {
    fn from(record: &RelayDevelopmentTransitionRecord) -> Self {
        Self {
            transition_digest: record.transition_digest(),
            partition: record.partition(),
            pre_state: record.pre_state().clone(),
            action: record.action(),
            post_state: record.post_state().clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayCalibrationTransitionRecord {
    world_digest: u64,
    transition_digest: u64,
    partition: CorpusPartition,
    profile_digest: u64,
    pre_state: PublicObservation,
    action: PublicAction,
    post_state: PublicObservation,
}

impl RelayCalibrationTransitionRecord {
    pub(super) fn world_digest(&self) -> u64 {
        self.world_digest
    }

    pub(super) fn transition_digest(&self) -> u64 {
        self.transition_digest
    }

    pub(super) fn partition(&self) -> CorpusPartition {
        self.partition
    }

    pub(super) fn profile_digest(&self) -> u64 {
        self.profile_digest
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

impl From<&RelayCalibrationTransitionRecord> for RelayBaselineRecord {
    fn from(record: &RelayCalibrationTransitionRecord) -> Self {
        Self {
            transition_digest: record.transition_digest,
            partition: record.partition,
            pre_state: record.pre_state.clone(),
            action: record.action,
            post_state: record.post_state.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct FittedRelayBaselines {
    fit_corpus_digest: u64,
    records: Vec<RelayBaselineRecord>,
}

impl FittedRelayBaselines {
    fn fit(development: &[RelayDevelopmentTransitionRecord]) -> Result<Self, RelayComparatorError> {
        if development.is_empty() {
            return Err(RelayComparatorError::EmptyDevelopmentCorpus);
        }
        if development
            .iter()
            .any(|record| record.partition() != CorpusPartition::Development)
        {
            return Err(RelayComparatorError::NonDevelopmentFitRecord);
        }
        let mut records: Vec<_> = development.iter().map(RelayBaselineRecord::from).collect();
        validate_unique_ids(&records)?;
        records.sort_by_key(|record| record.transition_digest);
        let fit_corpus_digest = corpus_digest(b"eureka.relay-fit-corpus.v1\0", &records);
        Ok(Self {
            fit_corpus_digest,
            records,
        })
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
        let matches: Vec<_> = self
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
        let candidates: Vec<_> = self
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
            output.push(choose_mode_transform(&counts)?.apply(pre.fields[index]));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RelayComparatorError {
    EmptyDevelopmentCorpus,
    NonDevelopmentFitRecord,
    DuplicateTransition,
    EmptyCalibrationProfiles,
    WrongCalibrationPartition,
    WrongScheduleRevision,
    NoScheduledAction,
    RealizedActionMismatch,
    MissingPostState,
    MissingTransitionIdentity,
    FitSelectionTransitionOverlap,
    Scoring(ConsequenceScoringError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayComparatorFreezeReceipt {
    pub runner_revision: &'static str,
    pub family_id: &'static str,
    pub schedule_revision: &'static str,
    pub development_receipt_digest: u64,
    pub learned_snapshot_replay_digest: u64,
    pub development_fit_corpus_digest: u64,
    pub calibration_profile_root: u64,
    pub calibration_transition_root: u64,
    pub calibration_selection_corpus_digest: u64,
    pub comparator_selection_replay_digest: u64,
    pub selection_status: PrimaryComparatorSelectionStatus,
    pub selected: Option<ShortcutBaselineKind>,
    pub calibration_world_count: u32,
    pub target_prediction_count: u32,
    pub replay_digest: u64,
}

#[derive(Debug, Clone)]
pub(super) struct RelayComparatorFreezeArtifact {
    pub fitted: FittedRelayBaselines,
    pub selection: PrimaryComparatorSelection,
    pub receipt: RelayComparatorFreezeReceipt,
    pub calibration_records: Vec<RelayCalibrationTransitionRecord>,
}

pub(super) fn freeze_relay_triad_comparator(
    development: &RelayDevelopmentArtifact,
) -> Result<RelayComparatorFreezeArtifact, RelayComparatorError> {
    let profiles = relay_scheduled_profiles(CorpusPartition::Calibration);
    debug_assert_eq!(profiles.len(), usize::from(RELAY_CALIBRATION_WORLDS));
    freeze_relay_triad_comparator_profiles(development, &profiles)
}

pub(super) fn freeze_relay_triad_comparator_profiles(
    development: &RelayDevelopmentArtifact,
    profiles: &[RelayTriadProfile],
) -> Result<RelayComparatorFreezeArtifact, RelayComparatorError> {
    if profiles.is_empty() {
        return Err(RelayComparatorError::EmptyCalibrationProfiles);
    }
    for profile in profiles {
        if profile.partition != CorpusPartition::Calibration {
            return Err(RelayComparatorError::WrongCalibrationPartition);
        }
        if profile.schedule_revision != RELAY_TRIAD_SCHEDULE_REVISION {
            return Err(RelayComparatorError::WrongScheduleRevision);
        }
    }

    let fitted = FittedRelayBaselines::fit(&development.records)?;
    let fit_ids: HashSet<u64> = development
        .records
        .iter()
        .map(RelayDevelopmentTransitionRecord::transition_digest)
        .collect();

    let mut calibration_records = Vec::with_capacity(profiles.len());
    let mut profile_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut transition_bytes = Vec::with_capacity(profiles.len() * 8);

    for profile in profiles {
        let profile_digest = profile.replay_digest();
        profile_bytes.extend_from_slice(&profile_digest.to_le_bytes());
        let mut evaluator = RelayTriadEvaluator::build(*profile);
        let pre_state = evaluator.runtime().observe();
        let legal = evaluator.runtime().legal_actions();
        let action = relay_scheduled_action(*profile, &legal)
            .ok_or(RelayComparatorError::NoScheduledAction)?;

        // Deliberately evaluator-only. No production FEP target call occurs.
        let receipt = evaluator.execute_qualified_action(action);
        if receipt.realized != Some(action) {
            return Err(RelayComparatorError::RealizedActionMismatch);
        }
        let post_state = receipt
            .post_state
            .ok_or(RelayComparatorError::MissingPostState)?;
        let transition_digest = receipt
            .transition_digest
            .ok_or(RelayComparatorError::MissingTransitionIdentity)?;
        if fit_ids.contains(&transition_digest) {
            return Err(RelayComparatorError::FitSelectionTransitionOverlap);
        }
        transition_bytes.extend_from_slice(&transition_digest.to_le_bytes());
        calibration_records.push(RelayCalibrationTransitionRecord {
            world_digest: receipt.world_digest,
            transition_digest,
            partition: CorpusPartition::Calibration,
            profile_digest,
            pre_state,
            action,
            post_state,
        });
    }

    let mut selection_records: Vec<RelayBaselineRecord> = calibration_records
        .iter()
        .map(RelayBaselineRecord::from)
        .collect();
    validate_unique_ids(&selection_records)?;
    selection_records.sort_by_key(|record| record.transition_digest);
    let selection_corpus_digest = corpus_digest(
        b"eureka.relay-calibration-selection-corpus.v1\0",
        &selection_records,
    );
    let selection = select_relay_primary_comparator(&fitted, &selection_records)?;

    let mut receipt = RelayComparatorFreezeReceipt {
        runner_revision: RELAY_COMPARATOR_FREEZE_REVISION,
        family_id: RELAY_TRIAD_FAMILY_ID,
        schedule_revision: RELAY_TRIAD_SCHEDULE_REVISION,
        development_receipt_digest: development.receipt.replay_digest,
        learned_snapshot_replay_digest: development.snapshot.replay_digest(),
        development_fit_corpus_digest: fitted.fit_corpus_digest(),
        calibration_profile_root: domain_hash(
            b"eureka.002o.relay-calibration-profiles.v1\0",
            &profile_bytes,
        ),
        calibration_transition_root: domain_hash(
            b"eureka.002o.relay-calibration-transitions.v1\0",
            &transition_bytes,
        ),
        calibration_selection_corpus_digest: selection_corpus_digest,
        comparator_selection_replay_digest: selection.replay_digest(),
        selection_status: selection.status,
        selected: selection.selected,
        calibration_world_count: profiles.len() as u32,
        target_prediction_count: RELAY_TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE,
        replay_digest: 0,
    };
    receipt.replay_digest = freeze_receipt_digest(&receipt);

    Ok(RelayComparatorFreezeArtifact {
        fitted,
        selection,
        receipt,
        calibration_records,
    })
}

fn select_relay_primary_comparator(
    fitted: &FittedRelayBaselines,
    calibration: &[RelayBaselineRecord],
) -> Result<PrimaryComparatorSelection, RelayComparatorError> {
    let mut receipts = Vec::with_capacity(EUREKA_002_ANALYSIS_PLAN_V1.eligible_comparators.len());
    for kind in EUREKA_002_ANALYSIS_PLAN_V1.eligible_comparators {
        receipts.push(calibrate_one(*kind, fitted, calibration)?);
    }
    receipts.sort_by(|a, b| a.kind.stable_id().cmp(b.kind.stable_id()));
    let selected = receipts
        .iter()
        .filter(|receipt| receipt.eligible)
        .copied()
        .reduce(|best, candidate| {
            if comparator_order(candidate, best) == Ordering::Greater {
                candidate
            } else {
                best
            }
        })
        .map(|receipt| receipt.kind);

    Ok(PrimaryComparatorSelection {
        analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
        fit_corpus_digest: fitted.fit_corpus_digest(),
        selection_corpus_digest: corpus_digest(
            b"eureka.relay-calibration-selection-corpus.v1\0",
            calibration,
        ),
        selection_revision: RELAY_COMPARATOR_SELECTION_REVISION,
        status: if selected.is_some() {
            PrimaryComparatorSelectionStatus::Selected
        } else {
            PrimaryComparatorSelectionStatus::InconclusiveNoEligibleComparator
        },
        selected,
        receipts,
    })
}

fn calibrate_one(
    kind: ShortcutBaselineKind,
    fitted: &FittedRelayBaselines,
    calibration: &[RelayBaselineRecord],
) -> Result<ComparatorCalibrationReceipt, RelayComparatorError> {
    let mut scored_trials = 0_u32;
    let mut abstained_trials = 0_u32;
    let mut out_of_domain_trials = 0_u32;
    let mut change_bearing_scored_trials = 0_u32;
    let mut tp = 0_u64;
    let mut fp = 0_u64;
    let mut missed = 0_u64;

    for transition in calibration {
        let prediction = fitted.predict(kind, &transition.pre_state, transition.action);
        let score = score_consequence(
            &transition.pre_state,
            &prediction,
            &transition.post_state,
        )
        .map_err(RelayComparatorError::Scoring)?;
        match score {
            ConsequenceScore::Scored(metrics) => {
                scored_trials += 1;
                if metrics.actual_changed > 0 {
                    change_bearing_scored_trials += 1;
                    tp += metrics.true_positive_changes as u64;
                    fp += metrics.false_positive_changes as u64;
                    missed += metrics.missed_changes as u64;
                }
            }
            ConsequenceScore::AbstainedInsufficientEvidence => abstained_trials += 1,
            ConsequenceScore::OutOfQualifiedDomain => out_of_domain_trials += 1,
        }
    }

    let total_trials = calibration.len() as u32;
    let coverage_bps = ((u64::from(scored_trials) * 10_000) / u64::from(total_trials)) as u16;
    let micro_f1_numerator = 2_u64.saturating_mul(tp);
    let micro_f1_denominator = micro_f1_numerator
        .saturating_add(fp)
        .saturating_add(missed);
    let eligible = coverage_bps >= EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
        && change_bearing_scored_trials > 0
        && micro_f1_denominator > 0;

    Ok(ComparatorCalibrationReceipt {
        kind,
        scored_trials,
        abstained_trials,
        out_of_domain_trials,
        total_trials,
        coverage_bps,
        change_bearing_scored_trials,
        true_positive_changes: tp,
        false_positive_changes: fp,
        missed_changes: missed,
        micro_f1_numerator,
        micro_f1_denominator,
        eligible,
    })
}

fn comparator_order(a: ComparatorCalibrationReceipt, b: ComparatorCalibrationReceipt) -> Ordering {
    let left = u128::from(a.micro_f1_numerator) * u128::from(b.micro_f1_denominator);
    let right = u128::from(b.micro_f1_numerator) * u128::from(a.micro_f1_denominator);
    left.cmp(&right)
        .then_with(|| b.kind.stable_id().cmp(a.kind.stable_id()))
}

fn validate_unique_ids(records: &[RelayBaselineRecord]) -> Result<(), RelayComparatorError> {
    let mut seen = HashSet::new();
    if records
        .iter()
        .any(|record| !seen.insert(record.transition_digest))
    {
        return Err(RelayComparatorError::DuplicateTransition);
    }
    Ok(())
}

fn corpus_digest(domain: &[u8], records: &[RelayBaselineRecord]) -> u64 {
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

fn freeze_receipt_digest(receipt: &RelayComparatorFreezeReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.runner_revision);
    encode_str(&mut bytes, receipt.family_id);
    encode_str(&mut bytes, receipt.schedule_revision);
    bytes.extend_from_slice(&receipt.development_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.learned_snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.development_fit_corpus_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.calibration_profile_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.calibration_transition_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.calibration_selection_corpus_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_selection_replay_digest.to_le_bytes());
    bytes.push(match receipt.selection_status {
        PrimaryComparatorSelectionStatus::Selected => 1,
        PrimaryComparatorSelectionStatus::InconclusiveNoEligibleComparator => 2,
    });
    match receipt.selected {
        Some(kind) => {
            bytes.push(1);
            encode_str(&mut bytes, kind.stable_id());
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&receipt.calibration_world_count.to_le_bytes());
    bytes.extend_from_slice(&receipt.target_prediction_count.to_le_bytes());
    fnv1a64(&bytes)
}

fn choose_modal_post(records: &[&RelayBaselineRecord]) -> Option<Vec<PublicValue>> {
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
            (PublicValue::Count(x), PublicValue::Count(y)) => i64::from(x).abs_diff(i64::from(y)),
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

fn encode_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn domain_hash(domain: &[u8], payload: &[u8]) -> u64 {
    let mut bytes = Vec::with_capacity(domain.len() + payload.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(payload);
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
    use crate::benchmarks::eureka::relay_triad_development::run_relay_triad_development_profiles;
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    fn development(n: usize) -> RelayDevelopmentArtifact {
        let profiles = relay_scheduled_profiles(CorpusPartition::Development);
        run_relay_triad_development_profiles(&service(), &profiles[..n]).unwrap()
    }

    fn calibration(n: usize) -> Vec<RelayTriadProfile> {
        relay_scheduled_profiles(CorpusPartition::Calibration)
            .into_iter()
            .take(n)
            .collect()
    }

    #[test]
    fn canonical_calibration_schedule_is_exact_128_worlds() {
        let profiles = relay_scheduled_profiles(CorpusPartition::Calibration);
        assert_eq!(profiles.len(), usize::from(RELAY_CALIBRATION_WORLDS));
    }

    #[test]
    fn comparator_freeze_consumes_zero_target_predictions() {
        let artifact = freeze_relay_triad_comparator_profiles(&development(12), &calibration(12))
            .unwrap();
        assert_eq!(artifact.receipt.target_prediction_count, 0);
        assert_eq!(artifact.receipt.calibration_world_count, 12);
        assert_eq!(artifact.calibration_records.len(), 12);
        assert_eq!(artifact.selection.receipts.len(), 4);
    }

    #[test]
    fn wrong_partition_rejects_before_calibration_execution() {
        let wrong = relay_scheduled_profiles(CorpusPartition::HeldOutEvaluation);
        assert!(matches!(
            freeze_relay_triad_comparator_profiles(&development(4), &wrong[..1]),
            Err(RelayComparatorError::WrongCalibrationPartition)
        ));
    }

    #[test]
    fn wrong_schedule_revision_rejects() {
        let mut profiles = calibration(1);
        profiles[0].schedule_revision = "not-relay-schedule";
        assert!(matches!(
            freeze_relay_triad_comparator_profiles(&development(4), &profiles),
            Err(RelayComparatorError::WrongScheduleRevision)
        ));
    }

    #[test]
    fn same_public_evidence_replays_same_selection_and_receipt() {
        let dev_profiles = relay_scheduled_profiles(CorpusPartition::Development);
        let dev_a = run_relay_triad_development_profiles(&service(), &dev_profiles[..16]).unwrap();
        let dev_b = run_relay_triad_development_profiles(&service(), &dev_profiles[..16]).unwrap();
        let profiles = calibration(16);
        let a = freeze_relay_triad_comparator_profiles(&dev_a, &profiles).unwrap();
        let b = freeze_relay_triad_comparator_profiles(&dev_b, &profiles).unwrap();
        assert_eq!(a.selection, b.selection);
        assert_eq!(a.receipt, b.receipt);
    }

    #[test]
    fn selection_kind_order_is_stable_and_complete() {
        let artifact = freeze_relay_triad_comparator_profiles(&development(16), &calibration(16))
            .unwrap();
        let ids: Vec<_> = artifact
            .selection
            .receipts
            .iter()
            .map(|receipt| receipt.kind.stable_id())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
        assert_eq!(ids.len(), EUREKA_002_ANALYSIS_PLAN_V1.eligible_comparators.len());
    }

    #[test]
    fn fitted_predictor_is_development_bound() {
        let dev = development(12);
        let artifact = freeze_relay_triad_comparator_profiles(&dev, &calibration(8)).unwrap();
        let prediction = artifact.fitted.predict(
            ShortcutBaselineKind::NearestTransition,
            dev.records[0].pre_state(),
            dev.records[0].action(),
        );
        assert_eq!(prediction.action, dev.records[0].action());
        assert_ne!(artifact.fitted.fit_corpus_digest(), 0);
    }
}
