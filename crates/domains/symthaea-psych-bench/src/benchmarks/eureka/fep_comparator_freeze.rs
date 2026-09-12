// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Target-blind ResourceFlow comparator freeze for EUREKA-002I.
//!
//! This module deliberately never constructs an FEP prediction session. It fits
//! shortcut controls on retained Development transitions, generates fresh
//! Calibration transitions with the evaluator alone, and freezes the primary
//! comparator before any target-held-out execution exists.

use super::baselines::{PublicTransitionRecord, TransitionRecordError, TransitionRecorder};
use super::campaign_manifest::{
    CAMPAIGN_SCHEDULE_REVISION, CampaignScheduleError, ScheduledWorldProfile, scheduled_action,
    scheduled_profiles,
};
use super::fep_development::FepDevelopmentArtifact;
use super::hidden_world::{CorpusPartition, FixtureFamily};
use super::selection::{
    ComparatorSelectionCorpus, ComparatorSelectionError, PrimaryComparatorSelection,
    PrimaryComparatorSelectionStatus, SelectionReadyFit, ShortcutBaselineKind,
    select_primary_comparator_v1,
};

pub(super) const RESOURCE_FLOW_COMPARATOR_FREEZE_REVISION: &str =
    "EUREKA.002I.RESOURCE_FLOW_COMPARATOR_FREEZE.v1";

/// Structural proof field: the comparator-freeze path has no target-prediction
/// input and therefore must always report zero target predictions.
pub(super) const TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE: u32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ComparatorFreezeError {
    EmptyCalibrationProfiles,
    WrongPartition,
    WrongFamily,
    Schedule(CampaignScheduleError),
    Transition(TransitionRecordError),
    Selection(ComparatorSelectionError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ComparatorFreezeReceipt {
    pub runner_revision: &'static str,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ComparatorFreezeArtifact {
    pub selection: PrimaryComparatorSelection,
    pub receipt: ComparatorFreezeReceipt,
    pub calibration_records: Vec<PublicTransitionRecord>,
}

/// Freeze the primary shortcut comparator using the canonical 128-world
/// ResourceFlow Calibration schedule. No target prediction is run here.
pub(super) fn freeze_resource_flow_comparator(
    development: &FepDevelopmentArtifact,
) -> Result<ComparatorFreezeArtifact, ComparatorFreezeError> {
    let profiles = scheduled_profiles(CorpusPartition::Calibration, FixtureFamily::ResourceFlow);
    freeze_resource_flow_comparator_profiles(development, &profiles)
}

/// Internal/testable executor over a Calibration subset. The complete profile
/// set is validated before the first evaluator transition is executed.
pub(super) fn freeze_resource_flow_comparator_profiles(
    development: &FepDevelopmentArtifact,
    profiles: &[ScheduledWorldProfile],
) -> Result<ComparatorFreezeArtifact, ComparatorFreezeError> {
    if profiles.is_empty() {
        return Err(ComparatorFreezeError::EmptyCalibrationProfiles);
    }
    for profile in profiles {
        if profile.partition != CorpusPartition::Calibration
            || profile.world.partition != CorpusPartition::Calibration
        {
            return Err(ComparatorFreezeError::WrongPartition);
        }
        if profile.family != FixtureFamily::ResourceFlow
            || profile.world.family != FixtureFamily::ResourceFlow
        {
            return Err(ComparatorFreezeError::WrongFamily);
        }
    }

    let fit = SelectionReadyFit::freeze(development.records.clone())
        .map_err(ComparatorFreezeError::Selection)?;
    let fit_corpus_digest = fit.fit_corpus_digest();

    let mut calibration_records = Vec::with_capacity(profiles.len());
    let mut profile_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut transition_bytes = Vec::with_capacity(profiles.len() * 8);

    for profile in profiles {
        profile_bytes.extend_from_slice(&profile.replay_digest().to_le_bytes());
        let mut recorder = TransitionRecorder::build(profile.world);
        let legal_actions = recorder.runtime().legal_actions();
        let public_action = scheduled_action(*profile, &legal_actions)
            .map_err(ComparatorFreezeError::Schedule)?;

        // Intentionally no FEP target/session call occurs on this path. The
        // evaluator alone realizes the frozen Calibration action schedule.
        let (_, record) = recorder
            .execute_and_record(public_action)
            .map_err(ComparatorFreezeError::Transition)?;
        transition_bytes.extend_from_slice(&record.transition_digest().to_le_bytes());
        calibration_records.push(record);
    }

    let selection_corpus = ComparatorSelectionCorpus::freeze(calibration_records.clone())
        .map_err(ComparatorFreezeError::Selection)?;
    let selection_corpus_digest = selection_corpus.digest();
    let selection = select_primary_comparator_v1(&fit, &selection_corpus)
        .map_err(ComparatorFreezeError::Selection)?;
    let selection_replay_digest = selection.replay_digest();

    let mut receipt = ComparatorFreezeReceipt {
        runner_revision: RESOURCE_FLOW_COMPARATOR_FREEZE_REVISION,
        schedule_revision: CAMPAIGN_SCHEDULE_REVISION,
        development_receipt_digest: development.receipt.replay_digest,
        learned_snapshot_replay_digest: development.snapshot.replay_digest(),
        development_fit_corpus_digest: fit_corpus_digest,
        calibration_profile_root: domain_hash(
            b"eureka.002i.calibration-profiles.v1\0",
            &profile_bytes,
        ),
        calibration_transition_root: domain_hash(
            b"eureka.002i.calibration-transitions.v1\0",
            &transition_bytes,
        ),
        calibration_selection_corpus_digest: selection_corpus_digest,
        comparator_selection_replay_digest: selection_replay_digest,
        selection_status: selection.status,
        selected: selection.selected,
        calibration_world_count: profiles.len() as u32,
        target_prediction_count: TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE,
        replay_digest: 0,
    };
    receipt.replay_digest = comparator_freeze_receipt_digest(&receipt);

    Ok(ComparatorFreezeArtifact {
        selection,
        receipt,
        calibration_records,
    })
}

fn comparator_freeze_receipt_digest(receipt: &ComparatorFreezeReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.runner_revision);
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
    use crate::benchmarks::eureka::fep_development::{
        run_resource_flow_development_profiles,
    };
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    fn development_subset(n: usize) -> FepDevelopmentArtifact {
        let profiles = scheduled_profiles(CorpusPartition::Development, FixtureFamily::ResourceFlow);
        run_resource_flow_development_profiles(&service(), &profiles[..n]).unwrap()
    }

    fn calibration_subset(n: usize) -> Vec<ScheduledWorldProfile> {
        scheduled_profiles(CorpusPartition::Calibration, FixtureFamily::ResourceFlow)
            .into_iter()
            .take(n)
            .collect()
    }

    #[test]
    fn canonical_calibration_schedule_is_exact_128_world_resource_flow() {
        let profiles = scheduled_profiles(CorpusPartition::Calibration, FixtureFamily::ResourceFlow);
        assert_eq!(profiles.len(), 128);
        assert!(profiles.iter().all(|profile| {
            profile.partition == CorpusPartition::Calibration
                && profile.family == FixtureFamily::ResourceFlow
        }));
    }

    #[test]
    fn comparator_freeze_consumes_zero_target_predictions() {
        let development = development_subset(9);
        let artifact = freeze_resource_flow_comparator_profiles(
            &development,
            &calibration_subset(9),
        )
        .unwrap();
        assert_eq!(artifact.receipt.target_prediction_count, 0);
        assert_eq!(artifact.receipt.calibration_world_count, 9);
        assert_eq!(artifact.calibration_records.len(), 9);
        assert!(artifact.calibration_records.iter().all(|record| {
            record.partition() == CorpusPartition::Calibration
                && record.family() == FixtureFamily::ResourceFlow
        }));
    }

    #[test]
    fn non_calibration_profiles_reject_before_selection_execution() {
        let development = development_subset(3);
        let wrong = scheduled_profiles(CorpusPartition::HeldOutEvaluation, FixtureFamily::ResourceFlow);
        assert!(matches!(
            freeze_resource_flow_comparator_profiles(&development, &wrong[..1]),
            Err(ComparatorFreezeError::WrongPartition)
        ));
    }

    #[test]
    fn causal_bits_profiles_reject_before_selection_execution() {
        let development = development_subset(3);
        let wrong = scheduled_profiles(CorpusPartition::Calibration, FixtureFamily::CausalBits);
        assert!(matches!(
            freeze_resource_flow_comparator_profiles(&development, &wrong[..1]),
            Err(ComparatorFreezeError::WrongFamily)
        ));
    }

    #[test]
    fn comparator_freeze_is_deterministic_for_fixed_public_data() {
        let development_a = development_subset(9);
        let development_b = development_subset(9);
        let profiles = calibration_subset(9);
        let a = freeze_resource_flow_comparator_profiles(&development_a, &profiles).unwrap();
        let b = freeze_resource_flow_comparator_profiles(&development_b, &profiles).unwrap();

        assert_eq!(a.receipt.development_fit_corpus_digest, b.receipt.development_fit_corpus_digest);
        assert_eq!(a.receipt.calibration_profile_root, b.receipt.calibration_profile_root);
        assert_eq!(a.receipt.calibration_transition_root, b.receipt.calibration_transition_root);
        assert_eq!(
            a.receipt.calibration_selection_corpus_digest,
            b.receipt.calibration_selection_corpus_digest
        );
        assert_eq!(
            a.receipt.comparator_selection_replay_digest,
            b.receipt.comparator_selection_replay_digest
        );
        assert_eq!(a.receipt.selection_status, b.receipt.selection_status);
        assert_eq!(a.receipt.selected, b.receipt.selected);
        assert_eq!(a.receipt.replay_digest, b.receipt.replay_digest);
    }

    #[test]
    fn selection_result_may_be_inconclusive_without_fallback() {
        let development = development_subset(1);
        let artifact = freeze_resource_flow_comparator_profiles(
            &development,
            &calibration_subset(1),
        )
        .unwrap();
        if artifact.selection.status
            == PrimaryComparatorSelectionStatus::InconclusiveNoEligibleComparator
        {
            assert_eq!(artifact.selection.selected, None);
            assert_eq!(artifact.receipt.selected, None);
        }
    }
}