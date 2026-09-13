// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen fitted comparator subject for RelayTriad HeldOut evaluation.
//!
//! The fitted object is cloned directly from the target-blind Calibration
//! artifact. There is no fitting/refitting operation in this module.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2261>

use super::baselines::ShortcutBaselineKind;
use super::consequence::ConsequencePrediction;
use super::hidden_world::{PublicAction, PublicObservation};
use super::relay_triad_comparator::{FittedRelayBaselines, RelayComparatorFreezeArtifact};
use super::relay_triad_heldout_seal::{
    RELAY_HELDOUT_OUTCOMES_AT_SEAL, RELAY_HELDOUT_TARGET_PREDICTIONS_AT_SEAL,
    RelayHeldOutAuthorization,
};
use super::selection::PrimaryComparatorSelectionStatus;

pub(super) const RELAY_FROZEN_COMPARATOR_SUBJECT_REVISION: &str =
    "EUREKA.002Q.RELAY_TRIAD_FROZEN_COMPARATOR.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RelayFrozenComparatorError {
    AuthorizationAlreadyUnsealed,
    ComparatorReceiptMismatch,
    ComparatorSelectionMismatch,
    SelectedComparatorMismatch,
    ComparatorNotSelected,
    DevelopmentReceiptMismatch,
    FitCorpusDigestMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayFrozenComparatorReceipt {
    pub revision: &'static str,
    pub authorization_replay_digest: u64,
    pub development_receipt_digest: u64,
    pub comparator_freeze_receipt_digest: u64,
    pub comparator_selection_replay_digest: u64,
    pub selected: ShortcutBaselineKind,
    pub fit_corpus_digest: u64,
    pub replay_digest: u64,
}

#[derive(Debug, Clone)]
pub(super) struct FrozenRelayHeldOutComparator {
    pub receipt: RelayFrozenComparatorReceipt,
    selected: ShortcutBaselineKind,
    fitted: FittedRelayBaselines,
}

impl FrozenRelayHeldOutComparator {
    pub(super) fn selected(&self) -> ShortcutBaselineKind {
        self.selected
    }

    pub(super) fn fit_corpus_digest(&self) -> u64 {
        self.fitted.fit_corpus_digest()
    }

    /// The complete future evaluation surface. No fit/refit API exists here.
    pub(super) fn predict(
        &self,
        pre: &PublicObservation,
        action: PublicAction,
    ) -> ConsequencePrediction {
        self.fitted.predict(self.selected, pre, action)
    }
}

pub(super) fn freeze_relay_heldout_comparator_subject(
    authorization: &RelayHeldOutAuthorization,
    comparator: &RelayComparatorFreezeArtifact,
) -> Result<FrozenRelayHeldOutComparator, RelayFrozenComparatorError> {
    let seal = &authorization.receipt;
    if seal.outcome_count_at_seal != RELAY_HELDOUT_OUTCOMES_AT_SEAL
        || seal.target_prediction_count_at_seal != RELAY_HELDOUT_TARGET_PREDICTIONS_AT_SEAL
    {
        return Err(RelayFrozenComparatorError::AuthorizationAlreadyUnsealed);
    }
    if seal.comparator_freeze_receipt_digest != comparator.receipt.replay_digest {
        return Err(RelayFrozenComparatorError::ComparatorReceiptMismatch);
    }
    if seal.comparator_selection_replay_digest != comparator.selection.replay_digest() {
        return Err(RelayFrozenComparatorError::ComparatorSelectionMismatch);
    }
    let selected = match (comparator.selection.status, comparator.selection.selected) {
        (PrimaryComparatorSelectionStatus::Selected, Some(kind)) => kind,
        _ => return Err(RelayFrozenComparatorError::ComparatorNotSelected),
    };
    if seal.selected_comparator != selected {
        return Err(RelayFrozenComparatorError::SelectedComparatorMismatch);
    }
    if seal.development_receipt_digest != comparator.receipt.development_receipt_digest {
        return Err(RelayFrozenComparatorError::DevelopmentReceiptMismatch);
    }
    let fit_corpus_digest = comparator.fitted.fit_corpus_digest();
    if fit_corpus_digest != comparator.receipt.development_fit_corpus_digest
        || fit_corpus_digest != comparator.selection.fit_corpus_digest
    {
        return Err(RelayFrozenComparatorError::FitCorpusDigestMismatch);
    }

    Ok(build_subject(
        seal.replay_digest,
        seal.development_receipt_digest,
        comparator.receipt.replay_digest,
        comparator.selection.replay_digest(),
        selected,
        comparator.fitted.clone(),
    ))
}

fn build_subject(
    authorization_replay_digest: u64,
    development_receipt_digest: u64,
    comparator_freeze_receipt_digest: u64,
    comparator_selection_replay_digest: u64,
    selected: ShortcutBaselineKind,
    fitted: FittedRelayBaselines,
) -> FrozenRelayHeldOutComparator {
    let mut receipt = RelayFrozenComparatorReceipt {
        revision: RELAY_FROZEN_COMPARATOR_SUBJECT_REVISION,
        authorization_replay_digest,
        development_receipt_digest,
        comparator_freeze_receipt_digest,
        comparator_selection_replay_digest,
        selected,
        fit_corpus_digest: fitted.fit_corpus_digest(),
        replay_digest: 0,
    };
    receipt.replay_digest = receipt_digest(&receipt);
    FrozenRelayHeldOutComparator {
        receipt,
        selected,
        fitted,
    }
}

fn receipt_digest(receipt: &RelayFrozenComparatorReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.revision);
    bytes.extend_from_slice(&receipt.authorization_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.development_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_freeze_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_selection_replay_digest.to_le_bytes());
    encode_str(&mut bytes, receipt.selected.stable_id());
    bytes.extend_from_slice(&receipt.fit_corpus_digest.to_le_bytes());
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
    use crate::benchmarks::eureka::hidden_world::CorpusPartition;
    use crate::benchmarks::eureka::relay_triad::relay_scheduled_profiles;
    use crate::benchmarks::eureka::relay_triad_comparator::freeze_relay_triad_comparator_profiles;
    use crate::benchmarks::eureka::relay_triad_development::run_relay_triad_development_profiles;
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    fn fitted_artifact() -> RelayComparatorFreezeArtifact {
        let dev_profiles = relay_scheduled_profiles(CorpusPartition::Development);
        let development =
            run_relay_triad_development_profiles(&service(), &dev_profiles[..12]).unwrap();
        let calibration = relay_scheduled_profiles(CorpusPartition::Calibration);
        freeze_relay_triad_comparator_profiles(&development, &calibration[..12]).unwrap()
    }

    #[test]
    fn subject_identity_binds_authorization_and_selected_kind() {
        let artifact = fitted_artifact();
        let selected = artifact
            .selection
            .selected
            .unwrap_or(ShortcutBaselineKind::NearestTransition);
        let a = build_subject(
            1,
            artifact.receipt.development_receipt_digest,
            artifact.receipt.replay_digest,
            artifact.selection.replay_digest(),
            selected,
            artifact.fitted.clone(),
        );
        let b = build_subject(
            2,
            artifact.receipt.development_receipt_digest,
            artifact.receipt.replay_digest,
            artifact.selection.replay_digest(),
            selected,
            artifact.fitted.clone(),
        );
        assert_ne!(a.receipt.replay_digest, b.receipt.replay_digest);

        let alternate = if selected == ShortcutBaselineKind::NearestTransition {
            ShortcutBaselineKind::SimpleMarkov
        } else {
            ShortcutBaselineKind::NearestTransition
        };
        let c = build_subject(
            1,
            artifact.receipt.development_receipt_digest,
            artifact.receipt.replay_digest,
            artifact.selection.replay_digest(),
            alternate,
            artifact.fitted.clone(),
        );
        assert_ne!(a.receipt.replay_digest, c.receipt.replay_digest);
    }

    #[test]
    fn frozen_subject_prediction_is_deterministic() {
        let artifact = fitted_artifact();
        let selected = artifact
            .selection
            .selected
            .unwrap_or(ShortcutBaselineKind::NearestTransition);
        let subject = build_subject(
            11,
            artifact.receipt.development_receipt_digest,
            artifact.receipt.replay_digest,
            artifact.selection.replay_digest(),
            selected,
            artifact.fitted.clone(),
        );
        let pre = artifact.calibration_records[0].pre_state().clone();
        let action = artifact.calibration_records[0].action();
        assert_eq!(subject.predict(&pre, action), subject.predict(&pre, action));
        assert_eq!(subject.fit_corpus_digest(), artifact.fitted.fit_corpus_digest());
        assert_eq!(subject.selected(), selected);
    }
}
