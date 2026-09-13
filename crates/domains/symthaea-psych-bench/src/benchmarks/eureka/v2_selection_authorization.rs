// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 V2 typed Calibration-only comparator-selection authorization.
//!
//! This source/test-only module freezes the evidence grammar and deterministic
//! selection theorem before a V2 Calibration runner exists. It does not execute
//! fitted baselines and therefore does not itself prove that the raw count
//! receipts came from real comparator execution. A future runner must produce
//! these receipts from the exact frozen subjects and canonical Calibration
//! corpus before this authorization becomes runtime evidence.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use super::analysis_plan::{ANALYSIS_PLAN_REVISION, EUREKA_002_ANALYSIS_PLAN_V1};
use super::baselines::ShortcutBaselineKind;
use super::hidden_world::PublicAction;
use super::v2_comparator_custody::{
    V2ComparatorCustodyError, V2ComparatorCustodyReceipt, V2CorpusPartition,
    V2DevelopmentFitCorpus, V2PublicTransitionEvidence, canonical_transition_semantics_bytes,
};
use super::v2_public_schema::{
    V2_OBSERVATION_DIM, V2PublicFamily, V2PublicState, public_schema_commitment,
};

pub(super) const V2_CALIBRATION_EVIDENCE_REVISION: &str =
    "EUREKA.002.V2.CALIBRATION_EVIDENCE.v2";
pub(super) const V2_CALIBRATION_CORPUS_REVISION: &str =
    "EUREKA.002.V2.CALIBRATION_CORPUS.v2";
pub(super) const V2_COMPARATOR_CALIBRATION_RECEIPT_REVISION: &str =
    "EUREKA.002.V2.COMPARATOR_CALIBRATION_RECEIPT.v2";
pub(super) const V2_SELECTION_IMPLEMENTATION_REVISION: &str =
    "EUREKA.002.V2.COMPARATOR_SELECTION_IMPLEMENTATION.v1";
pub(super) const V2_SELECTION_AUTHORIZATION_REVISION: &str =
    "EUREKA.002.V2.COMPARATOR_SELECTION_AUTHORIZATION.v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2SelectionAuthorizationError {
    CanonicalEvidence(V2ComparatorCustodyError),
    EmptyCalibrationCorpus,
    DevelopmentCalibrationIdentityOverlap,
    DevelopmentCalibrationTransitionOverlap,
    DuplicateCalibrationRowIdentity,
    DuplicateCalibrationTransition,
    StatusCountMismatch,
    ChangeBearingCountExceedsScored,
    ChangeBearingCountsInconsistent,
    ChangedFieldCountsExceedCapacity,
    MissingComparatorReceipt,
    DuplicateComparatorReceipt,
    UnexpectedComparatorReceipt,
    ReceiptTrialCountMismatch,
    FitCorpusCommitmentMismatch,
    Custody(V2ComparatorCustodyError),
}

impl From<V2ComparatorCustodyError> for V2SelectionAuthorizationError {
    fn from(value: V2ComparatorCustodyError) -> Self {
        Self::CanonicalEvidence(value)
    }
}

/// Canonical Calibration transition evidence.
///
/// The constructor delegates public-schema/family/action/context admissibility
/// to the same canonical transition type used by Development custody. The
/// transition semantic key deliberately excludes partition and row identity so
/// exact public-transition reuse can be detected across Development/Calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2CalibrationEvidence {
    row_identity: [u8; 32],
    transition_semantics_bytes: Vec<u8>,
    canonical_bytes: Vec<u8>,
}

impl V2CalibrationEvidence {
    pub(super) fn new(
        family: V2PublicFamily,
        row_identity: [u8; 32],
        pre: V2PublicState,
        action: PublicAction,
        post: V2PublicState,
    ) -> Result<Self, V2SelectionAuthorizationError> {
        // Reuse the exact canonical admissibility theorem. Calibration is fixed
        // by construction and cannot be caller-relabeled as HeldOut/External.
        let validated = V2PublicTransitionEvidence::new(
            family,
            V2CorpusPartition::Calibration,
            row_identity,
            pre,
            action,
            post,
        )?;
        let transition_semantics_bytes = canonical_transition_semantics_bytes(&validated);

        let mut canonical_bytes = Vec::new();
        encode_bytes(
            &mut canonical_bytes,
            V2_CALIBRATION_EVIDENCE_REVISION.as_bytes(),
        );
        canonical_bytes.extend_from_slice(&public_schema_commitment());
        canonical_bytes.push(2); // canonical V2 Calibration partition tag
        canonical_bytes.extend_from_slice(&row_identity);
        canonical_bytes.extend_from_slice(&transition_semantics_bytes);

        Ok(Self {
            row_identity,
            transition_semantics_bytes,
            canonical_bytes,
        })
    }

    pub(super) const fn row_identity(&self) -> [u8; 32] {
        self.row_identity
    }
}

/// Canonically ordered Calibration-only corpus bound to one exact Development
/// fit corpus. Both identity and exact public-transition semantic disjointness
/// are checked before any selection decision is minted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2CalibrationCorpus {
    records: Vec<V2CalibrationEvidence>,
    schema_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2CalibrationCorpus {
    pub(super) fn freeze(
        development: &V2DevelopmentFitCorpus,
        mut records: Vec<V2CalibrationEvidence>,
    ) -> Result<Self, V2SelectionAuthorizationError> {
        if records.is_empty() {
            return Err(V2SelectionAuthorizationError::EmptyCalibrationCorpus);
        }

        let development_ids: BTreeSet<[u8; 32]> = development
            .records()
            .iter()
            .map(V2PublicTransitionEvidence::row_identity)
            .collect();
        let development_transitions: BTreeSet<Vec<u8>> = development
            .records()
            .iter()
            .map(canonical_transition_semantics_bytes)
            .collect();
        let mut seen_ids = BTreeSet::new();
        let mut seen_transitions = BTreeSet::new();
        for record in &records {
            if development_ids.contains(&record.row_identity) {
                return Err(
                    V2SelectionAuthorizationError::DevelopmentCalibrationIdentityOverlap,
                );
            }
            if development_transitions.contains(&record.transition_semantics_bytes) {
                return Err(
                    V2SelectionAuthorizationError::DevelopmentCalibrationTransitionOverlap,
                );
            }
            if !seen_ids.insert(record.row_identity) {
                return Err(V2SelectionAuthorizationError::DuplicateCalibrationRowIdentity);
            }
            if !seen_transitions.insert(record.transition_semantics_bytes.clone()) {
                return Err(V2SelectionAuthorizationError::DuplicateCalibrationTransition);
            }
        }
        records.sort_by_key(|record| record.row_identity);

        let schema_commitment = public_schema_commitment();
        let fit_corpus_commitment = development.commitment();
        let commitment = calibration_corpus_commitment(
            schema_commitment,
            fit_corpus_commitment,
            &records,
        );
        Ok(Self {
            records,
            schema_commitment,
            fit_corpus_commitment,
            commitment,
        })
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

    pub(super) fn len(&self) -> usize {
        self.records.len()
    }
}

/// Raw integer result of evaluating one preregistered comparator over the exact
/// Calibration corpus. Coverage, F1 fraction, and eligibility are derived here;
/// they are never caller-supplied fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2ComparatorCalibrationReceipt {
    kind: ShortcutBaselineKind,
    total_trials: u32,
    scored_trials: u32,
    abstained_trials: u32,
    out_of_domain_trials: u32,
    coverage_bps: u16,
    change_bearing_scored_trials: u32,
    true_positive_changes: u64,
    false_positive_changes: u64,
    missed_changes: u64,
    micro_f1_numerator: u64,
    micro_f1_denominator: u64,
    eligible: bool,
}

impl V2ComparatorCalibrationReceipt {
    /// Private on purpose: until a real V2 Calibration runner exists, only this
    /// module's regression corpus may synthesize raw counts. The eventual runner
    /// should live behind this module or add a constructor that consumes scorer
    /// output directly rather than exposing an arbitrary public receipt mint.
    fn from_raw_counts(
        kind: ShortcutBaselineKind,
        total_trials: u32,
        scored_trials: u32,
        abstained_trials: u32,
        out_of_domain_trials: u32,
        change_bearing_scored_trials: u32,
        true_positive_changes: u64,
        false_positive_changes: u64,
        missed_changes: u64,
    ) -> Result<Self, V2SelectionAuthorizationError> {
        let status_total = scored_trials
            .saturating_add(abstained_trials)
            .saturating_add(out_of_domain_trials);
        if total_trials == 0 || status_total != total_trials {
            return Err(V2SelectionAuthorizationError::StatusCountMismatch);
        }
        if change_bearing_scored_trials > scored_trials {
            return Err(V2SelectionAuthorizationError::ChangeBearingCountExceedsScored);
        }

        let actual_changed_fields = true_positive_changes.saturating_add(missed_changes);
        let max_actual_changed_fields = u64::from(change_bearing_scored_trials)
            .saturating_mul(u64::try_from(V2_OBSERVATION_DIM).expect("V2 dimension fits u64"));
        let change_bearing_consistent = if change_bearing_scored_trials == 0 {
            actual_changed_fields == 0
        } else {
            actual_changed_fields >= u64::from(change_bearing_scored_trials)
                && actual_changed_fields <= max_actual_changed_fields
        };
        if !change_bearing_consistent {
            return Err(V2SelectionAuthorizationError::ChangeBearingCountsInconsistent);
        }

        let changed_event_total = actual_changed_fields.saturating_add(false_positive_changes);
        let max_events = u64::from(scored_trials)
            .saturating_mul(u64::try_from(V2_OBSERVATION_DIM).expect("V2 dimension fits u64"));
        if changed_event_total > max_events {
            return Err(V2SelectionAuthorizationError::ChangedFieldCountsExceedCapacity);
        }

        let coverage_bps = u16::try_from(
            (u64::from(scored_trials) * 10_000_u64) / u64::from(total_trials),
        )
        .expect("coverage basis points fit u16");
        let micro_f1_numerator = 2_u64.saturating_mul(true_positive_changes);
        let micro_f1_denominator = micro_f1_numerator
            .saturating_add(false_positive_changes)
            .saturating_add(missed_changes);
        let eligible = coverage_bps >= EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
            && change_bearing_scored_trials > 0
            && micro_f1_denominator > 0;

        Ok(Self {
            kind,
            total_trials,
            scored_trials,
            abstained_trials,
            out_of_domain_trials,
            coverage_bps,
            change_bearing_scored_trials,
            true_positive_changes,
            false_positive_changes,
            missed_changes,
            micro_f1_numerator,
            micro_f1_denominator,
            eligible,
        })
    }

    pub(super) const fn kind(self) -> ShortcutBaselineKind {
        self.kind
    }

    pub(super) const fn eligible(self) -> bool {
        self.eligible
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2ComparatorSelectionAuthorization {
    schema_commitment: [u8; 32],
    analysis_plan_replay_digest: u64,
    analysis_plan_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    selected: ShortcutBaselineKind,
    receipts: Vec<V2ComparatorCalibrationReceipt>,
    commitment: [u8; 32],
}

impl V2ComparatorSelectionAuthorization {
    pub(super) const fn schema_commitment(&self) -> [u8; 32] {
        self.schema_commitment
    }

    pub(super) const fn fit_corpus_commitment(&self) -> [u8; 32] {
        self.fit_corpus_commitment
    }

    pub(super) const fn calibration_corpus_commitment(&self) -> [u8; 32] {
        self.calibration_corpus_commitment
    }

    pub(super) const fn selected(&self) -> ShortcutBaselineKind {
        self.selected
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2InconclusiveSelectionReceipt {
    schema_commitment: [u8; 32],
    analysis_plan_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    receipts: Vec<V2ComparatorCalibrationReceipt>,
    commitment: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum V2ComparatorSelectionOutcome {
    Selected(V2ComparatorSelectionAuthorization),
    Inconclusive(V2InconclusiveSelectionReceipt),
}

/// Deterministically reconstruct the preregistered comparator choice from a
/// complete set of raw-count receipts. No selected kind or eligibility bit is
/// accepted from the caller.
pub(super) fn authorize_selection(
    development: &V2DevelopmentFitCorpus,
    calibration: &V2CalibrationCorpus,
    mut receipts: Vec<V2ComparatorCalibrationReceipt>,
) -> Result<V2ComparatorSelectionOutcome, V2SelectionAuthorizationError> {
    if development.commitment() != calibration.fit_corpus_commitment {
        return Err(V2SelectionAuthorizationError::FitCorpusCommitmentMismatch);
    }

    receipts.sort_by(|left, right| left.kind.stable_id().cmp(right.kind.stable_id()));
    validate_complete_receipt_set(&receipts, calibration.len())?;

    let selected = receipts
        .iter()
        .copied()
        .filter(|receipt| receipt.eligible)
        .reduce(|best, candidate| {
            if comparator_receipt_order(candidate, best) == Ordering::Greater {
                candidate
            } else {
                best
            }
        })
        .map(|receipt| receipt.kind);

    let schema_commitment = calibration.schema_commitment;
    let analysis_plan_replay_digest = EUREKA_002_ANALYSIS_PLAN_V1.replay_digest();
    let analysis_plan_commitment = EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment();
    let fit_corpus_commitment = development.commitment();
    let calibration_corpus_commitment = calibration.commitment;

    match selected {
        Some(selected) => {
            let commitment = selection_commitment(
                1,
                schema_commitment,
                analysis_plan_replay_digest,
                analysis_plan_commitment,
                fit_corpus_commitment,
                calibration_corpus_commitment,
                Some(selected),
                &receipts,
            );
            Ok(V2ComparatorSelectionOutcome::Selected(
                V2ComparatorSelectionAuthorization {
                    schema_commitment,
                    analysis_plan_replay_digest,
                    analysis_plan_commitment,
                    fit_corpus_commitment,
                    calibration_corpus_commitment,
                    selected,
                    receipts,
                    commitment,
                },
            ))
        }
        None => {
            let commitment = selection_commitment(
                2,
                schema_commitment,
                analysis_plan_replay_digest,
                analysis_plan_commitment,
                fit_corpus_commitment,
                calibration_corpus_commitment,
                None,
                &receipts,
            );
            Ok(V2ComparatorSelectionOutcome::Inconclusive(
                V2InconclusiveSelectionReceipt {
                    schema_commitment,
                    analysis_plan_commitment,
                    fit_corpus_commitment,
                    calibration_corpus_commitment,
                    receipts,
                    commitment,
                },
            ))
        }
    }
}

/// Typed bridge into comparator custody. Inconclusive selection has no type that
/// can call this function, so it cannot unlock a comparator subject.
pub(super) fn freeze_comparator_custody_from_authorization(
    development: &V2DevelopmentFitCorpus,
    authorization: &V2ComparatorSelectionAuthorization,
) -> Result<V2ComparatorCustodyReceipt, V2SelectionAuthorizationError> {
    if development.commitment() != authorization.fit_corpus_commitment {
        return Err(V2SelectionAuthorizationError::FitCorpusCommitmentMismatch);
    }
    V2ComparatorCustodyReceipt::freeze(
        development,
        authorization.selected,
        authorization.commitment,
    )
    .map_err(V2SelectionAuthorizationError::Custody)
}

fn validate_complete_receipt_set(
    receipts: &[V2ComparatorCalibrationReceipt],
    calibration_len: usize,
) -> Result<(), V2SelectionAuthorizationError> {
    let plan = EUREKA_002_ANALYSIS_PLAN_V1;
    let expected_total = u32::try_from(calibration_len).expect("V2 Calibration corpus fits u32");
    if receipts.len() < plan.eligible_comparators.len() {
        return Err(V2SelectionAuthorizationError::MissingComparatorReceipt);
    }
    if receipts.len() > plan.eligible_comparators.len() {
        return Err(V2SelectionAuthorizationError::UnexpectedComparatorReceipt);
    }

    let mut seen = BTreeSet::new();
    for receipt in receipts {
        let stable_id = receipt.kind.stable_id();
        if !plan
            .eligible_comparators
            .iter()
            .any(|kind| kind.stable_id() == stable_id)
        {
            return Err(V2SelectionAuthorizationError::UnexpectedComparatorReceipt);
        }
        if !seen.insert(stable_id) {
            return Err(V2SelectionAuthorizationError::DuplicateComparatorReceipt);
        }
        if receipt.total_trials != expected_total {
            return Err(V2SelectionAuthorizationError::ReceiptTrialCountMismatch);
        }
    }
    if plan
        .eligible_comparators
        .iter()
        .any(|kind| !seen.contains(kind.stable_id()))
    {
        return Err(V2SelectionAuthorizationError::MissingComparatorReceipt);
    }
    Ok(())
}

/// Ordering where Greater means `left` is the preferred comparator. Exact
/// cross-multiplication avoids rounded-score tie ambiguity. Equal fractions use
/// the preregistered stable-ID order (lexicographically smaller ID wins).
fn comparator_receipt_order(
    left: V2ComparatorCalibrationReceipt,
    right: V2ComparatorCalibrationReceipt,
) -> Ordering {
    let left_value = u128::from(left.micro_f1_numerator)
        .saturating_mul(u128::from(right.micro_f1_denominator));
    let right_value = u128::from(right.micro_f1_numerator)
        .saturating_mul(u128::from(left.micro_f1_denominator));
    left_value
        .cmp(&right_value)
        .then_with(|| right.kind.stable_id().cmp(left.kind.stable_id()))
}

fn calibration_corpus_commitment(
    schema_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    records: &[V2CalibrationEvidence],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_CALIBRATION_CORPUS_REVISION.as_bytes());
    bytes.extend_from_slice(&schema_commitment);
    bytes.extend_from_slice(&fit_corpus_commitment);
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for record in records {
        encode_bytes(&mut bytes, &record.canonical_bytes);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn selection_commitment(
    status_tag: u8,
    schema_commitment: [u8; 32],
    analysis_plan_replay_digest: u64,
    analysis_plan_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    calibration_corpus_commitment: [u8; 32],
    selected: Option<ShortcutBaselineKind>,
    receipts: &[V2ComparatorCalibrationReceipt],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_SELECTION_AUTHORIZATION_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_SELECTION_IMPLEMENTATION_REVISION.as_bytes());
    encode_bytes(&mut bytes, ANALYSIS_PLAN_REVISION.as_bytes());
    bytes.extend_from_slice(&analysis_plan_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&analysis_plan_commitment);
    bytes.extend_from_slice(&schema_commitment);
    bytes.extend_from_slice(&fit_corpus_commitment);
    bytes.extend_from_slice(&calibration_corpus_commitment);
    bytes.push(status_tag);
    match selected {
        Some(kind) => {
            bytes.push(1);
            encode_bytes(&mut bytes, kind.stable_id().as_bytes());
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&(receipts.len() as u64).to_le_bytes());
    for receipt in receipts {
        encode_receipt(&mut bytes, receipt);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn encode_receipt(bytes: &mut Vec<u8>, receipt: &V2ComparatorCalibrationReceipt) {
    encode_bytes(
        bytes,
        V2_COMPARATOR_CALIBRATION_RECEIPT_REVISION.as_bytes(),
    );
    encode_bytes(bytes, receipt.kind.stable_id().as_bytes());
    bytes.extend_from_slice(&receipt.total_trials.to_le_bytes());
    bytes.extend_from_slice(&receipt.scored_trials.to_le_bytes());
    bytes.extend_from_slice(&receipt.abstained_trials.to_le_bytes());
    bytes.extend_from_slice(&receipt.out_of_domain_trials.to_le_bytes());
    bytes.extend_from_slice(&receipt.coverage_bps.to_le_bytes());
    bytes.extend_from_slice(&receipt.change_bearing_scored_trials.to_le_bytes());
    bytes.extend_from_slice(&receipt.true_positive_changes.to_le_bytes());
    bytes.extend_from_slice(&receipt.false_positive_changes.to_le_bytes());
    bytes.extend_from_slice(&receipt.missed_changes.to_le_bytes());
    bytes.extend_from_slice(&receipt.micro_f1_numerator.to_le_bytes());
    bytes.extend_from_slice(&receipt.micro_f1_denominator.to_le_bytes());
    bytes.push(u8::from(receipt.eligible));
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn development_record(id: u8, offset: i32) -> V2PublicTransitionEvidence {
        let mut row_identity = [0_u8; 32];
        row_identity[0] = id;
        V2PublicTransitionEvidence::new(
            V2PublicFamily::PublicFlowV2,
            V2CorpusPartition::Development,
            row_identity,
            V2PublicState::new([3 + offset, 4, 5, 0]).unwrap(),
            PublicAction::Pulse { slot: 0 },
            V2PublicState::new([2 + offset, 5, 5, 0]).unwrap(),
        )
        .unwrap()
    }

    fn development() -> V2DevelopmentFitCorpus {
        V2DevelopmentFitCorpus::freeze(vec![
            development_record(1, 0),
            development_record(2, 1),
        ])
        .unwrap()
    }

    fn calibration_record(id: u8, offset: i32) -> V2CalibrationEvidence {
        let mut row_identity = [0_u8; 32];
        row_identity[0] = id;
        V2CalibrationEvidence::new(
            V2PublicFamily::PublicFlowV2,
            row_identity,
            V2PublicState::new([8 + offset, 3, 2, 0]).unwrap(),
            PublicAction::Pulse { slot: 1 },
            V2PublicState::new([8 + offset, 2, 3, 0]).unwrap(),
        )
        .unwrap()
    }

    fn calibration(development: &V2DevelopmentFitCorpus) -> V2CalibrationCorpus {
        V2CalibrationCorpus::freeze(
            development,
            vec![calibration_record(11, 0), calibration_record(12, 1)],
        )
        .unwrap()
    }

    fn receipt(
        kind: ShortcutBaselineKind,
        tp: u64,
        fp: u64,
        missed: u64,
    ) -> V2ComparatorCalibrationReceipt {
        V2ComparatorCalibrationReceipt::from_raw_counts(
            kind, 2, 2, 0, 0, 2, tp, fp, missed,
        )
        .unwrap()
    }

    fn complete_receipts() -> Vec<V2ComparatorCalibrationReceipt> {
        vec![
            receipt(ShortcutBaselineKind::ActionMarginalDelta, 4, 1, 1),
            receipt(ShortcutBaselineKind::ExactLookup, 1, 3, 3),
            receipt(ShortcutBaselineKind::NearestTransition, 5, 0, 1),
            receipt(ShortcutBaselineKind::SimpleMarkov, 3, 2, 2),
        ]
    }

    #[test]
    fn calibration_corpus_commitment_is_order_invariant() {
        let development = development();
        let forward = V2CalibrationCorpus::freeze(
            &development,
            vec![calibration_record(11, 0), calibration_record(12, 1)],
        )
        .unwrap();
        let reverse = V2CalibrationCorpus::freeze(
            &development,
            vec![calibration_record(12, 1), calibration_record(11, 0)],
        )
        .unwrap();
        assert_eq!(forward.commitment(), reverse.commitment());
        assert_eq!(forward.fit_corpus_commitment(), development.commitment());
        assert_eq!(forward.schema_commitment(), public_schema_commitment());
    }

    #[test]
    fn development_calibration_identity_overlap_fails_closed() {
        let development = development();
        let mut row_identity = [0_u8; 32];
        row_identity[0] = 1;
        let overlap = V2CalibrationEvidence::new(
            V2PublicFamily::PublicFlowV2,
            row_identity,
            V2PublicState::new([9, 3, 2, 0]).unwrap(),
            PublicAction::NoOp,
            V2PublicState::new([9, 3, 2, 0]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            V2CalibrationCorpus::freeze(&development, vec![overlap]),
            Err(V2SelectionAuthorizationError::DevelopmentCalibrationIdentityOverlap)
        );
    }

    #[test]
    fn development_calibration_semantic_overlap_under_different_id_fails_closed() {
        let development = development();
        let mut row_identity = [0_u8; 32];
        row_identity[0] = 99;
        let overlap = V2CalibrationEvidence::new(
            V2PublicFamily::PublicFlowV2,
            row_identity,
            V2PublicState::new([3, 4, 5, 0]).unwrap(),
            PublicAction::Pulse { slot: 0 },
            V2PublicState::new([2, 5, 5, 0]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            V2CalibrationCorpus::freeze(&development, vec![overlap]),
            Err(V2SelectionAuthorizationError::DevelopmentCalibrationTransitionOverlap)
        );
    }

    #[test]
    fn duplicate_calibration_transition_under_different_id_fails_closed() {
        let development = development();
        let a = calibration_record(11, 0);
        let mut b = a.clone();
        b.row_identity = [0xA5_u8; 32];
        let mut canonical = Vec::new();
        encode_bytes(&mut canonical, V2_CALIBRATION_EVIDENCE_REVISION.as_bytes());
        canonical.extend_from_slice(&public_schema_commitment());
        canonical.push(2);
        canonical.extend_from_slice(&b.row_identity);
        canonical.extend_from_slice(&b.transition_semantics_bytes);
        b.canonical_bytes = canonical;
        assert_eq!(
            V2CalibrationCorpus::freeze(&development, vec![a, b]),
            Err(V2SelectionAuthorizationError::DuplicateCalibrationTransition)
        );
    }

    #[test]
    fn raw_receipt_derives_coverage_f1_and_eligibility() {
        let eligible = V2ComparatorCalibrationReceipt::from_raw_counts(
            ShortcutBaselineKind::NearestTransition,
            10,
            9,
            1,
            0,
            8,
            12,
            2,
            2,
        )
        .unwrap();
        assert_eq!(eligible.coverage_bps, 9_000);
        assert_eq!(eligible.micro_f1_numerator, 24);
        assert_eq!(eligible.micro_f1_denominator, 28);
        assert!(eligible.eligible());

        assert_eq!(
            V2ComparatorCalibrationReceipt::from_raw_counts(
                ShortcutBaselineKind::NearestTransition,
                10,
                9,
                0,
                0,
                8,
                1,
                1,
                1,
            ),
            Err(V2SelectionAuthorizationError::StatusCountMismatch)
        );
        assert_eq!(
            V2ComparatorCalibrationReceipt::from_raw_counts(
                ShortcutBaselineKind::NearestTransition,
                2,
                2,
                0,
                0,
                0,
                1,
                0,
                0,
            ),
            Err(V2SelectionAuthorizationError::ChangeBearingCountsInconsistent)
        );
    }

    #[test]
    fn exact_fraction_selection_is_order_invariant() {
        let development = development();
        let calibration = calibration(&development);
        let forward = authorize_selection(&development, &calibration, complete_receipts()).unwrap();
        let mut reverse_receipts = complete_receipts();
        reverse_receipts.reverse();
        let reverse = authorize_selection(&development, &calibration, reverse_receipts).unwrap();
        let (V2ComparatorSelectionOutcome::Selected(forward), V2ComparatorSelectionOutcome::Selected(reverse)) = (forward, reverse) else {
            panic!("fixture must select comparator");
        };
        assert_eq!(forward.selected(), ShortcutBaselineKind::NearestTransition);
        assert_eq!(forward.commitment(), reverse.commitment());
        assert_eq!(forward.calibration_corpus_commitment(), calibration.commitment());
    }

    #[test]
    fn exact_f1_tie_uses_stable_id_not_input_order() {
        let development = development();
        let calibration = calibration(&development);
        let tied = vec![
            receipt(ShortcutBaselineKind::SimpleMarkov, 1, 1, 1),
            receipt(ShortcutBaselineKind::NearestTransition, 1, 1, 1),
            receipt(ShortcutBaselineKind::ExactLookup, 1, 1, 1),
            receipt(ShortcutBaselineKind::ActionMarginalDelta, 1, 1, 1),
        ];
        let outcome = authorize_selection(&development, &calibration, tied).unwrap();
        let V2ComparatorSelectionOutcome::Selected(authorization) = outcome else {
            panic!("all four tied receipts are eligible");
        };
        assert_eq!(
            authorization.selected(),
            ShortcutBaselineKind::ActionMarginalDelta
        );
    }

    #[test]
    fn incomplete_or_duplicate_receipt_set_fails_closed() {
        let development = development();
        let calibration = calibration(&development);
        let mut missing = complete_receipts();
        missing.pop();
        assert_eq!(
            authorize_selection(&development, &calibration, missing),
            Err(V2SelectionAuthorizationError::MissingComparatorReceipt)
        );

        let mut duplicate = complete_receipts();
        duplicate[3] = duplicate[0];
        assert_eq!(
            authorize_selection(&development, &calibration, duplicate),
            Err(V2SelectionAuthorizationError::DuplicateComparatorReceipt)
        );
    }

    #[test]
    fn changing_raw_counts_changes_selection_authorization_commitment() {
        let development = development();
        let calibration = calibration(&development);
        let first = authorize_selection(&development, &calibration, complete_receipts()).unwrap();
        let mut changed = complete_receipts();
        changed[0] = receipt(ShortcutBaselineKind::ActionMarginalDelta, 3, 2, 1);
        let second = authorize_selection(&development, &calibration, changed).unwrap();
        let (V2ComparatorSelectionOutcome::Selected(first), V2ComparatorSelectionOutcome::Selected(second)) = (first, second) else {
            panic!("fixtures must remain selected");
        };
        assert_ne!(first.commitment(), second.commitment());
    }

    #[test]
    fn no_eligible_comparator_is_inconclusive_not_authorized() {
        let development = development();
        let calibration = calibration(&development);
        let receipts = EUREKA_002_ANALYSIS_PLAN_V1
            .eligible_comparators
            .iter()
            .copied()
            .map(|kind| {
                V2ComparatorCalibrationReceipt::from_raw_counts(
                    kind, 2, 1, 1, 0, 1, 1, 0, 0,
                )
                .unwrap()
            })
            .collect();
        let outcome = authorize_selection(&development, &calibration, receipts).unwrap();
        assert!(matches!(outcome, V2ComparatorSelectionOutcome::Inconclusive(_)));
    }

    #[test]
    fn typed_selected_authorization_can_mint_custody_for_exact_fit_only() {
        let development = development();
        let calibration = calibration(&development);
        let outcome = authorize_selection(&development, &calibration, complete_receipts()).unwrap();
        let V2ComparatorSelectionOutcome::Selected(authorization) = outcome else {
            panic!("fixture must select comparator");
        };
        let custody = freeze_comparator_custody_from_authorization(&development, &authorization)
            .unwrap();
        assert_eq!(custody.selected(), authorization.selected());
        assert_eq!(custody.fit_corpus_commitment(), development.commitment());
        assert_ne!(custody.commitment(), [0_u8; 32]);

        let other = V2DevelopmentFitCorpus::freeze(vec![development_record(9, 4)]).unwrap();
        assert_eq!(
            freeze_comparator_custody_from_authorization(&other, &authorization),
            Err(V2SelectionAuthorizationError::FitCorpusCommitmentMismatch)
        );
    }
}
