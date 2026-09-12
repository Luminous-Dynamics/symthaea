// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic calibration-only primary comparator selection for EUREKA-002D.
//!
//! Baseline fitting is Development-only on this canonical path. Comparator
//! selection is Calibration-only and cannot inspect target HeldOutEvaluation.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2101>

use std::cmp::Ordering;
use std::collections::HashSet;

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::baselines::{
    BaselineFitCorpus, FittedShortcutBaselines, PublicTransitionRecord, ShortcutBaselineKind,
};
use super::consequence::{ConsequenceScore, ConsequenceScoringError, score_consequence};
use super::hidden_world::CorpusPartition;

pub(super) const COMPARATOR_SELECTION_REVISION: &str =
    "EUREKA.002D.CALIBRATION_COMPARATOR_SELECTION.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ComparatorSelectionError {
    EmptyFitCorpus,
    NonDevelopmentFitRecord,
    FitCorpusRejected,
    EmptySelectionCorpus,
    NonCalibrationSelectionRecord,
    DuplicateSelectionTransition,
    FitSelectionTransitionOverlap,
    Scoring(ConsequenceScoringError),
}

/// Canonical fit object for comparator selection. Unlike the lower-level
/// baseline primitive, this path reserves Development exclusively for fitting.
pub(super) struct SelectionReadyFit {
    fitted: FittedShortcutBaselines,
    fit_corpus_digest: u64,
    fit_transition_ids: HashSet<u64>,
}

impl SelectionReadyFit {
    pub(super) fn freeze(
        records: Vec<PublicTransitionRecord>,
    ) -> Result<Self, ComparatorSelectionError> {
        if records.is_empty() {
            return Err(ComparatorSelectionError::EmptyFitCorpus);
        }
        if records
            .iter()
            .any(|record| record.partition() != CorpusPartition::Development)
        {
            return Err(ComparatorSelectionError::NonDevelopmentFitRecord);
        }
        let fit_transition_ids = records
            .iter()
            .map(PublicTransitionRecord::transition_digest)
            .collect();
        let corpus = BaselineFitCorpus::freeze(records)
            .map_err(|_| ComparatorSelectionError::FitCorpusRejected)?;
        let fit_corpus_digest = corpus.digest();
        let fitted = FittedShortcutBaselines::fit(&corpus);
        Ok(Self {
            fitted,
            fit_corpus_digest,
            fit_transition_ids,
        })
    }

    pub(super) fn fit_corpus_digest(&self) -> u64 {
        self.fit_corpus_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ComparatorSelectionCorpus {
    records: Vec<PublicTransitionRecord>,
    digest: u64,
}

impl ComparatorSelectionCorpus {
    pub(super) fn freeze(
        mut records: Vec<PublicTransitionRecord>,
    ) -> Result<Self, ComparatorSelectionError> {
        if records.is_empty() {
            return Err(ComparatorSelectionError::EmptySelectionCorpus);
        }
        if records
            .iter()
            .any(|record| record.partition() != CorpusPartition::Calibration)
        {
            return Err(ComparatorSelectionError::NonCalibrationSelectionRecord);
        }
        let mut seen = HashSet::new();
        for record in &records {
            if !seen.insert(record.transition_digest()) {
                return Err(ComparatorSelectionError::DuplicateSelectionTransition);
            }
        }
        records.sort_by_key(PublicTransitionRecord::transition_digest);
        let digest = selection_corpus_digest(&records);
        Ok(Self { records, digest })
    }

    pub(super) fn digest(&self) -> u64 {
        self.digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ComparatorCalibrationReceipt {
    pub kind: ShortcutBaselineKind,
    pub scored_trials: u32,
    pub abstained_trials: u32,
    pub out_of_domain_trials: u32,
    pub total_trials: u32,
    pub coverage_bps: u16,
    pub change_bearing_scored_trials: u32,
    pub true_positive_changes: u64,
    pub false_positive_changes: u64,
    pub missed_changes: u64,
    pub micro_f1_numerator: u64,
    pub micro_f1_denominator: u64,
    pub eligible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrimaryComparatorSelectionStatus {
    Selected,
    InconclusiveNoEligibleComparator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PrimaryComparatorSelection {
    pub analysis_plan_digest: u64,
    pub fit_corpus_digest: u64,
    pub selection_corpus_digest: u64,
    pub selection_revision: &'static str,
    pub status: PrimaryComparatorSelectionStatus,
    pub selected: Option<ShortcutBaselineKind>,
    pub receipts: Vec<ComparatorCalibrationReceipt>,
}

impl PrimaryComparatorSelection {
    pub(super) fn replay_digest(&self) -> u64 {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"eureka.primary-comparator-selection.v1\0");
        bytes.extend_from_slice(&self.analysis_plan_digest.to_le_bytes());
        bytes.extend_from_slice(&self.fit_corpus_digest.to_le_bytes());
        bytes.extend_from_slice(&self.selection_corpus_digest.to_le_bytes());
        encode_str(&mut bytes, self.selection_revision);
        bytes.push(match self.status {
            PrimaryComparatorSelectionStatus::Selected => 1,
            PrimaryComparatorSelectionStatus::InconclusiveNoEligibleComparator => 2,
        });
        match self.selected {
            Some(kind) => {
                bytes.push(1);
                encode_str(&mut bytes, kind.stable_id());
            }
            None => bytes.push(0),
        }
        bytes.extend_from_slice(&(self.receipts.len() as u64).to_le_bytes());
        for receipt in &self.receipts {
            encode_str(&mut bytes, receipt.kind.stable_id());
            bytes.extend_from_slice(&receipt.scored_trials.to_le_bytes());
            bytes.extend_from_slice(&receipt.abstained_trials.to_le_bytes());
            bytes.extend_from_slice(&receipt.out_of_domain_trials.to_le_bytes());
            bytes.extend_from_slice(&receipt.total_trials.to_le_bytes());
            bytes.extend_from_slice(&receipt.coverage_bps.to_le_bytes());
            bytes.extend_from_slice(&receipt.change_bearing_scored_trials.to_le_bytes());
            bytes.extend_from_slice(&receipt.true_positive_changes.to_le_bytes());
            bytes.extend_from_slice(&receipt.false_positive_changes.to_le_bytes());
            bytes.extend_from_slice(&receipt.missed_changes.to_le_bytes());
            bytes.extend_from_slice(&receipt.micro_f1_numerator.to_le_bytes());
            bytes.extend_from_slice(&receipt.micro_f1_denominator.to_le_bytes());
            bytes.push(u8::from(receipt.eligible));
        }
        fnv1a64(&bytes)
    }
}

pub(super) fn select_primary_comparator_v1(
    fit: &SelectionReadyFit,
    selection: &ComparatorSelectionCorpus,
) -> Result<PrimaryComparatorSelection, ComparatorSelectionError> {
    if selection
        .records
        .iter()
        .any(|record| fit.fit_transition_ids.contains(&record.transition_digest()))
    {
        return Err(ComparatorSelectionError::FitSelectionTransitionOverlap);
    }

    let plan = EUREKA_002_ANALYSIS_PLAN_V1;
    let mut receipts = Vec::with_capacity(plan.eligible_comparators.len());
    for kind in plan.eligible_comparators {
        receipts.push(calibrate_one(*kind, fit, selection)?);
    }
    receipts.sort_by(|a, b| a.kind.stable_id().cmp(b.kind.stable_id()));

    let selected = receipts
        .iter()
        .filter(|receipt| receipt.eligible)
        .copied()
        .reduce(|best, candidate| {
            if comparator_receipt_order(candidate, best) == Ordering::Greater {
                candidate
            } else {
                best
            }
        })
        .map(|receipt| receipt.kind);

    Ok(PrimaryComparatorSelection {
        analysis_plan_digest: plan.replay_digest(),
        fit_corpus_digest: fit.fit_corpus_digest,
        selection_corpus_digest: selection.digest,
        selection_revision: COMPARATOR_SELECTION_REVISION,
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
    fit: &SelectionReadyFit,
    selection: &ComparatorSelectionCorpus,
) -> Result<ComparatorCalibrationReceipt, ComparatorSelectionError> {
    let mut scored_trials = 0_u32;
    let mut abstained_trials = 0_u32;
    let mut out_of_domain_trials = 0_u32;
    let mut change_bearing_scored_trials = 0_u32;
    let mut tp = 0_u64;
    let mut fp = 0_u64;
    let mut missed = 0_u64;

    for transition in &selection.records {
        let prediction = fit
            .fitted
            .predict(kind, transition.pre_state(), transition.action());
        let score = score_consequence(
            transition.pre_state(),
            &prediction,
            transition.post_state(),
        )
        .map_err(ComparatorSelectionError::Scoring)?;
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

    let total_trials = selection.records.len() as u32;
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

/// Ordering where Greater means `a` is the preferred comparator.
fn comparator_receipt_order(
    a: ComparatorCalibrationReceipt,
    b: ComparatorCalibrationReceipt,
) -> Ordering {
    let left = u128::from(a.micro_f1_numerator) * u128::from(b.micro_f1_denominator);
    let right = u128::from(b.micro_f1_numerator) * u128::from(a.micro_f1_denominator);
    left.cmp(&right).then_with(|| b.kind.stable_id().cmp(a.kind.stable_id()))
}

fn selection_corpus_digest(records: &[PublicTransitionRecord]) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.comparator-selection-corpus.v1\0");
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for record in records {
        bytes.extend_from_slice(&record.transition_digest().to_le_bytes());
    }
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
    use crate::benchmarks::eureka::hidden_world::{
        FixtureFamily, PublicAction, WorldBuildProfile,
    };

    fn capture(seed: u64, partition: CorpusPartition, action: PublicAction) -> PublicTransitionRecord {
        let mut recorder = TransitionRecorder::build(WorldBuildProfile {
            family: FixtureFamily::CausalBits,
            seed,
            mechanism_variant: 0,
            partition,
        });
        recorder.execute_and_record(action).unwrap().1
    }

    fn fit() -> SelectionReadyFit {
        SelectionReadyFit::freeze(vec![
            capture(1, CorpusPartition::Development, PublicAction::NoOp),
            capture(2, CorpusPartition::Development, PublicAction::NoOp),
            capture(3, CorpusPartition::Development, PublicAction::Pulse { slot: 0 }),
            capture(4, CorpusPartition::Development, PublicAction::Pulse { slot: 1 }),
        ])
        .unwrap()
    }

    fn selection() -> ComparatorSelectionCorpus {
        ComparatorSelectionCorpus::freeze(vec![
            capture(101, CorpusPartition::Calibration, PublicAction::NoOp),
            capture(102, CorpusPartition::Calibration, PublicAction::NoOp),
            capture(103, CorpusPartition::Calibration, PublicAction::Pulse { slot: 0 }),
            capture(104, CorpusPartition::Calibration, PublicAction::Pulse { slot: 1 }),
        ])
        .unwrap()
    }

    #[test]
    fn canonical_fit_path_is_development_only() {
        let calibration = capture(1, CorpusPartition::Calibration, PublicAction::NoOp);
        assert!(matches!(
            SelectionReadyFit::freeze(vec![calibration]),
            Err(ComparatorSelectionError::NonDevelopmentFitRecord)
        ));
    }

    #[test]
    fn selection_corpus_is_calibration_only() {
        let held_out = capture(10, CorpusPartition::HeldOutEvaluation, PublicAction::NoOp);
        assert_eq!(
            ComparatorSelectionCorpus::freeze(vec![held_out]),
            Err(ComparatorSelectionError::NonCalibrationSelectionRecord)
        );
    }

    #[test]
    fn selection_corpus_identity_is_order_invariant() {
        let a = capture(101, CorpusPartition::Calibration, PublicAction::NoOp);
        let b = capture(102, CorpusPartition::Calibration, PublicAction::NoOp);
        let ab = ComparatorSelectionCorpus::freeze(vec![a.clone(), b.clone()]).unwrap();
        let ba = ComparatorSelectionCorpus::freeze(vec![b, a]).unwrap();
        assert_eq!(ab.digest(), ba.digest());
    }

    #[test]
    fn overlap_fails_closed_even_if_injected_into_internal_test_fixture() {
        let mut fit = fit();
        let selection = selection();
        fit.fit_transition_ids
            .insert(selection.records[0].transition_digest());
        assert_eq!(
            select_primary_comparator_v1(&fit, &selection),
            Err(ComparatorSelectionError::FitSelectionTransitionOverlap)
        );
    }

    #[test]
    fn micro_f1_comparison_uses_exact_integer_ratio() {
        let a = ComparatorCalibrationReceipt {
            kind: ShortcutBaselineKind::NearestTransition,
            scored_trials: 10,
            abstained_trials: 0,
            out_of_domain_trials: 0,
            total_trials: 10,
            coverage_bps: 10_000,
            change_bearing_scored_trials: 10,
            true_positive_changes: 7,
            false_positive_changes: 2,
            missed_changes: 1,
            micro_f1_numerator: 14,
            micro_f1_denominator: 17,
            eligible: true,
        };
        let b = ComparatorCalibrationReceipt {
            micro_f1_numerator: 8,
            micro_f1_denominator: 10,
            kind: ShortcutBaselineKind::SimpleMarkov,
            ..a
        };
        assert_eq!(comparator_receipt_order(a, b), Ordering::Greater);
    }

    #[test]
    fn exact_ratio_tie_breaks_by_stable_id_ascending() {
        let a = ComparatorCalibrationReceipt {
            kind: ShortcutBaselineKind::ActionMarginalDelta,
            scored_trials: 10,
            abstained_trials: 0,
            out_of_domain_trials: 0,
            total_trials: 10,
            coverage_bps: 10_000,
            change_bearing_scored_trials: 5,
            true_positive_changes: 2,
            false_positive_changes: 1,
            missed_changes: 1,
            micro_f1_numerator: 4,
            micro_f1_denominator: 6,
            eligible: true,
        };
        let b = ComparatorCalibrationReceipt {
            kind: ShortcutBaselineKind::ExactLookup,
            ..a
        };
        assert_eq!(comparator_receipt_order(a, b), Ordering::Greater);
    }

    #[test]
    fn selection_receipt_is_deterministic() {
        let fit = fit();
        let selection = selection();
        let a = select_primary_comparator_v1(&fit, &selection).unwrap();
        let b = select_primary_comparator_v1(&fit, &selection).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.replay_digest(), b.replay_digest());
        assert_eq!(a.fit_corpus_digest, fit.fit_corpus_digest());
        assert_eq!(a.selection_corpus_digest, selection.digest());
    }

    #[test]
    fn every_receipt_retains_coverage_and_raw_change_counts() {
        let result = select_primary_comparator_v1(&fit(), &selection()).unwrap();
        assert_eq!(result.receipts.len(), EUREKA_002_ANALYSIS_PLAN_V1.eligible_comparators.len());
        for receipt in result.receipts {
            assert_eq!(
                receipt.scored_trials + receipt.abstained_trials + receipt.out_of_domain_trials,
                receipt.total_trials
            );
        }
    }

    #[test]
    fn no_eligible_comparator_is_explicitly_inconclusive() {
        let fit = SelectionReadyFit::freeze(vec![capture(
            1,
            CorpusPartition::Development,
            PublicAction::NoOp,
        )])
        .unwrap();
        let selection = ComparatorSelectionCorpus::freeze(vec![capture(
            101,
            CorpusPartition::Calibration,
            PublicAction::Pulse { slot: 3 },
        )])
        .unwrap();
        let result = select_primary_comparator_v1(&fit, &selection).unwrap();
        assert_eq!(
            result.status,
            PrimaryComparatorSelectionStatus::InconclusiveNoEligibleComparator
        );
        assert_eq!(result.selected, None);
    }
}
