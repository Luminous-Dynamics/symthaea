// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen fitted comparator subject for EUREKA-002K.
//!
//! Comparator selection and comparator fitting are separate scientific facts.
//! #2155 freezes which shortcut baseline wins; this module additionally freezes
//! the exact fitted baseline predictor that a future HeldOut runner may query.

use super::baselines::{
    BaselineFitCorpus, CorpusError, FittedShortcutBaselines, PublicTransitionRecord,
    ShortcutBaselineKind,
};
use super::consequence::ConsequencePrediction;
use super::fep_comparator_freeze::ComparatorFreezeArtifact;
use super::fep_development::FepDevelopmentArtifact;
use super::heldout_seal::{HeldOutSealReceipt, HELDOUT_OUTCOMES_AT_SEAL, HELDOUT_TARGET_PREDICTIONS_AT_SEAL};
use super::hidden_world::{CorpusPartition, FixtureFamily, PublicAction, PublicObservation};
use super::selection::PrimaryComparatorSelectionStatus;

pub(super) const FROZEN_COMPARATOR_SUBJECT_REVISION: &str =
    "EUREKA.002K.FROZEN_COMPARATOR_SUBJECT.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FrozenComparatorError {
    AuthorizationAlreadyUnsealed,
    AuthorizationDevelopmentMismatch,
    AuthorizationComparatorMismatch,
    AuthorizationSelectionMismatch,
    AuthorizationSelectedComparatorMismatch,
    ComparatorNotSelected,
    DevelopmentRecordMismatch,
    FitCorpus(CorpusError),
    FitCorpusDigestMismatch,
    FittedSubjectDigestMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FrozenComparatorReceipt {
    pub revision: &'static str,
    pub authorization_replay_digest: u64,
    pub development_receipt_digest: u64,
    pub comparator_freeze_receipt_digest: u64,
    pub comparator_selection_replay_digest: u64,
    pub selected: ShortcutBaselineKind,
    pub fit_corpus_digest: u64,
    pub development_record_count: u32,
    pub replay_digest: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FrozenHeldOutComparator {
    pub receipt: FrozenComparatorReceipt,
    selected: ShortcutBaselineKind,
    fitted: FittedShortcutBaselines,
}

impl FrozenHeldOutComparator {
    pub(super) fn selected(&self) -> ShortcutBaselineKind {
        self.selected
    }

    pub(super) fn fit_corpus_digest(&self) -> u64 {
        self.fitted.fit_corpus_digest()
    }

    /// The only prediction surface a future HeldOut runner needs.
    pub(super) fn predict(
        &self,
        pre: &PublicObservation,
        action: PublicAction,
    ) -> ConsequencePrediction {
        self.fitted.predict(self.selected, pre, action)
    }
}

/// Freeze the exact fitted comparator predictor already selected before
/// HeldOut.  This function consumes Development records only; Calibration is
/// used solely for the pre-existing selected-kind proof in #2155.
pub(super) fn freeze_selected_heldout_comparator(
    authorization: &HeldOutSealReceipt,
    development: &FepDevelopmentArtifact,
    comparator: &ComparatorFreezeArtifact,
) -> Result<FrozenHeldOutComparator, FrozenComparatorError> {
    if authorization.outcome_count_at_seal != HELDOUT_OUTCOMES_AT_SEAL
        || authorization.target_prediction_count_at_seal
            != HELDOUT_TARGET_PREDICTIONS_AT_SEAL
    {
        return Err(FrozenComparatorError::AuthorizationAlreadyUnsealed);
    }
    if authorization.development_receipt_digest != development.receipt.replay_digest {
        return Err(FrozenComparatorError::AuthorizationDevelopmentMismatch);
    }
    if authorization.comparator_freeze_receipt_digest != comparator.receipt.replay_digest {
        return Err(FrozenComparatorError::AuthorizationComparatorMismatch);
    }
    if authorization.comparator_selection_replay_digest != comparator.selection.replay_digest() {
        return Err(FrozenComparatorError::AuthorizationSelectionMismatch);
    }

    let selected = match (comparator.selection.status, comparator.selection.selected) {
        (PrimaryComparatorSelectionStatus::Selected, Some(kind)) => kind,
        _ => return Err(FrozenComparatorError::ComparatorNotSelected),
    };
    if authorization.selected_comparator != selected {
        return Err(FrozenComparatorError::AuthorizationSelectedComparatorMismatch);
    }

    if development.records.is_empty()
        || development.records.iter().any(|record| {
            record.partition() != CorpusPartition::Development
                || record.family() != FixtureFamily::ResourceFlow
        })
    {
        return Err(FrozenComparatorError::DevelopmentRecordMismatch);
    }

    let corpus = BaselineFitCorpus::freeze(development.records.clone())
        .map_err(FrozenComparatorError::FitCorpus)?;
    if corpus.digest() != comparator.receipt.development_fit_corpus_digest {
        return Err(FrozenComparatorError::FitCorpusDigestMismatch);
    }
    let fitted = FittedShortcutBaselines::fit(&corpus);
    if fitted.fit_corpus_digest() != corpus.digest() {
        return Err(FrozenComparatorError::FittedSubjectDigestMismatch);
    }

    Ok(build_frozen_comparator(
        authorization.replay_digest,
        development.receipt.replay_digest,
        comparator.receipt.replay_digest,
        comparator.selection.replay_digest(),
        selected,
        fitted,
        development.records.len() as u32,
    ))
}

fn build_frozen_comparator(
    authorization_replay_digest: u64,
    development_receipt_digest: u64,
    comparator_freeze_receipt_digest: u64,
    comparator_selection_replay_digest: u64,
    selected: ShortcutBaselineKind,
    fitted: FittedShortcutBaselines,
    development_record_count: u32,
) -> FrozenHeldOutComparator {
    let mut receipt = FrozenComparatorReceipt {
        revision: FROZEN_COMPARATOR_SUBJECT_REVISION,
        authorization_replay_digest,
        development_receipt_digest,
        comparator_freeze_receipt_digest,
        comparator_selection_replay_digest,
        selected,
        fit_corpus_digest: fitted.fit_corpus_digest(),
        development_record_count,
        replay_digest: 0,
    };
    receipt.replay_digest = frozen_comparator_receipt_digest(&receipt);
    FrozenHeldOutComparator {
        receipt,
        selected,
        fitted,
    }
}

fn frozen_comparator_receipt_digest(receipt: &FrozenComparatorReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.revision);
    bytes.extend_from_slice(&receipt.authorization_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.development_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_freeze_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_selection_replay_digest.to_le_bytes());
    encode_str(&mut bytes, receipt.selected.stable_id());
    bytes.extend_from_slice(&receipt.fit_corpus_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.development_record_count.to_le_bytes());
    fnv1a64(&bytes)
}

fn encode_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
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
    use crate::benchmarks::eureka::baselines::TransitionRecorder;
    use crate::benchmarks::eureka::hidden_world::WorldBuildProfile;

    fn development_records() -> Vec<PublicTransitionRecord> {
        let actions = [
            PublicAction::NoOp,
            PublicAction::Transfer {
                from: 0,
                to: 1,
                amount: 1,
            },
            PublicAction::Transfer {
                from: 1,
                to: 2,
                amount: 1,
            },
        ];
        (0..9_u64)
            .map(|index| {
                let mut recorder = TransitionRecorder::build(WorldBuildProfile {
                    family: FixtureFamily::ResourceFlow,
                    seed: 0x1000_0000_0000_1000 + index,
                    mechanism_variant: (index % 2) as u8,
                    partition: CorpusPartition::Development,
                });
                recorder
                    .execute_and_record(actions[index as usize % actions.len()])
                    .unwrap()
                    .1
            })
            .collect()
    }

    fn fitted() -> FittedShortcutBaselines {
        let corpus = BaselineFitCorpus::freeze(development_records()).unwrap();
        FittedShortcutBaselines::fit(&corpus)
    }

    #[test]
    fn frozen_subject_prediction_is_deterministic() {
        let records = development_records();
        let probe_pre = records[0].pre_state().clone();
        let probe_action = records[0].action();
        let a = build_frozen_comparator(
            11,
            12,
            13,
            14,
            ShortcutBaselineKind::NearestTransition,
            fitted(),
            9,
        );
        let b = build_frozen_comparator(
            11,
            12,
            13,
            14,
            ShortcutBaselineKind::NearestTransition,
            fitted(),
            9,
        );
        assert_eq!(a.receipt, b.receipt);
        assert_eq!(a.predict(&probe_pre, probe_action), b.predict(&probe_pre, probe_action));
    }

    #[test]
    fn authorization_identity_is_part_of_comparator_subject() {
        let a = build_frozen_comparator(
            21,
            22,
            23,
            24,
            ShortcutBaselineKind::SimpleMarkov,
            fitted(),
            9,
        );
        let b = build_frozen_comparator(
            99,
            22,
            23,
            24,
            ShortcutBaselineKind::SimpleMarkov,
            fitted(),
            9,
        );
        assert_ne!(a.receipt.replay_digest, b.receipt.replay_digest);
    }

    #[test]
    fn selected_kind_is_part_of_comparator_subject() {
        let a = build_frozen_comparator(
            31,
            32,
            33,
            34,
            ShortcutBaselineKind::NearestTransition,
            fitted(),
            9,
        );
        let b = build_frozen_comparator(
            31,
            32,
            33,
            34,
            ShortcutBaselineKind::ActionMarginalDelta,
            fitted(),
            9,
        );
        assert_ne!(a.receipt.replay_digest, b.receipt.replay_digest);
    }

    #[test]
    fn fit_corpus_identity_is_part_of_comparator_subject() {
        let mut records = development_records();
        let corpus_a = BaselineFitCorpus::freeze(records.clone()).unwrap();
        records.pop();
        let corpus_b = BaselineFitCorpus::freeze(records).unwrap();
        let a = build_frozen_comparator(
            41,
            42,
            43,
            44,
            ShortcutBaselineKind::NearestTransition,
            FittedShortcutBaselines::fit(&corpus_a),
            9,
        );
        let b = build_frozen_comparator(
            41,
            42,
            43,
            44,
            ShortcutBaselineKind::NearestTransition,
            FittedShortcutBaselines::fit(&corpus_b),
            8,
        );
        assert_ne!(a.fit_corpus_digest(), b.fit_corpus_digest());
        assert_ne!(a.receipt.replay_digest, b.receipt.replay_digest);
    }
}
