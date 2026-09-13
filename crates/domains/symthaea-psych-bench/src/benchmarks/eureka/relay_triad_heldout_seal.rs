// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Zero-outcome HeldOut authorization for the RelayTriad sidecar family.
//!
//! This module revalidates RelayTriad Development/Calibration lineage and then
//! freezes exact HeldOut profiles, evaluator identities, public actions and
//! mapped target action indices. It never executes a HeldOut transition or asks
//! the target for a prediction.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2252>

use symthaea_fep::FepEvaluationSnapshot;

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::baselines::ShortcutBaselineKind;
use super::campaign_manifest::{CandidateBindingError, CandidateSubjectBindingV1};
use super::hidden_world::{CorpusPartition, PublicAction};
use super::relay_triad::{
    RELAY_HELD_OUT_WORLDS, RELAY_TRIAD_FAMILY_ID, RELAY_TRIAD_SCHEDULE_REVISION,
    RELAY_TRIAD_TARGET_ADAPTER_REVISION, RelayTriadEvaluator, RelayTriadFepTargetContract,
    RelayTriadProfile, RelayTriadTargetContractError, relay_scheduled_action,
    relay_scheduled_profiles,
};
use super::relay_triad_comparator::{
    RELAY_COMPARATOR_FREEZE_REVISION, RELAY_TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE,
    RelayComparatorFreezeArtifact,
};
use super::relay_triad_development::{
    RELAY_DEVELOPMENT_RUNNER_REVISION, RelayDevelopmentArtifact,
    RelayDevelopmentReceipt, RelayDevelopmentTransitionRecord,
};
use super::selection::PrimaryComparatorSelectionStatus;
use super::target_contract::EurekaTargetScope;

pub(super) const RELAY_HELDOUT_SEAL_REVISION: &str =
    "EUREKA.002P.RELAY_TRIAD_HELDOUT_SEAL.v1";
pub(super) const RELAY_HELDOUT_OUTCOMES_AT_SEAL: u32 = 0;
pub(super) const RELAY_HELDOUT_TARGET_PREDICTIONS_AT_SEAL: u32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RelayHeldOutSealError {
    InvalidLegacyCandidate(CandidateBindingError),
    AdapterRevisionMismatch,
    DevelopmentIdentityMismatch,
    DevelopmentCountMismatch,
    DevelopmentRecordMismatch,
    DevelopmentProfileRootMismatch,
    DevelopmentTransitionRootMismatch,
    DevelopmentReceiptMismatch,
    ComparatorIdentityMismatch,
    ComparatorCountMismatch,
    ComparatorTargetLeakage,
    CalibrationRecordMismatch,
    CalibrationProfileRootMismatch,
    CalibrationTransitionRootMismatch,
    FitCorpusDigestMismatch,
    SelectionCorpusDigestMismatch,
    SelectionIdentityMismatch,
    ComparatorNotSelected,
    TargetContract(RelayTriadTargetContractError),
    HeldOutScheduleMismatch,
    HeldOutNoLegalAction,
    HeldOutTargetActionMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayScopedCandidateBinding {
    legacy_candidate_digest: u64,
    scope: EurekaTargetScope,
    family_id: &'static str,
    snapshot_replay_digest: u64,
    target_contract_digest: u64,
    adapter_revision: &'static str,
    replay_digest: u64,
}

impl RelayScopedCandidateBinding {
    fn build(
        legacy: &CandidateSubjectBindingV1,
        snapshot: &FepEvaluationSnapshot,
        contract: &RelayTriadFepTargetContract,
    ) -> Result<Self, RelayHeldOutSealError> {
        let legacy_candidate_digest = legacy
            .replay_digest()
            .map_err(RelayHeldOutSealError::InvalidLegacyCandidate)?;
        if legacy.target_adapter_revision != RELAY_TRIAD_TARGET_ADAPTER_REVISION {
            return Err(RelayHeldOutSealError::AdapterRevisionMismatch);
        }
        if contract.snapshot_replay_digest() != snapshot.replay_digest() {
            return Err(RelayHeldOutSealError::DevelopmentIdentityMismatch);
        }
        let mut binding = Self {
            legacy_candidate_digest,
            scope: EurekaTargetScope::ProductionFepComponentSnapshot,
            family_id: RELAY_TRIAD_FAMILY_ID,
            snapshot_replay_digest: snapshot.replay_digest(),
            target_contract_digest: contract.replay_digest(),
            adapter_revision: RELAY_TRIAD_TARGET_ADAPTER_REVISION,
            replay_digest: 0,
        };
        binding.replay_digest = scoped_candidate_digest(&binding);
        Ok(binding)
    }

    pub(super) fn replay_digest(&self) -> u64 {
        self.replay_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayHeldOutSealReceipt {
    pub revision: &'static str,
    pub analysis_plan_digest: u64,
    pub legacy_candidate_digest: u64,
    pub scoped_candidate_digest: u64,
    pub learned_snapshot_replay_digest: u64,
    pub target_contract_digest: u64,
    pub target_scope: EurekaTargetScope,
    pub family_id: &'static str,
    pub development_receipt_digest: u64,
    pub comparator_freeze_receipt_digest: u64,
    pub comparator_selection_replay_digest: u64,
    pub selected_comparator: ShortcutBaselineKind,
    pub schedule_revision: &'static str,
    pub heldout_profile_root: u64,
    pub heldout_world_root: u64,
    pub heldout_public_action_root: u64,
    pub heldout_target_action_root: u64,
    pub heldout_world_count: u32,
    pub outcome_count_at_seal: u32,
    pub target_prediction_count_at_seal: u32,
    pub replay_digest: u64,
}

#[derive(Debug, Clone)]
pub(super) struct RelayHeldOutAuthorization {
    pub receipt: RelayHeldOutSealReceipt,
    pub snapshot: FepEvaluationSnapshot,
    pub contract: RelayTriadFepTargetContract,
    pub scoped_candidate: RelayScopedCandidateBinding,
    pub profiles: Vec<RelayTriadProfile>,
    pub public_actions: Vec<PublicAction>,
    pub target_actions: Vec<usize>,
}

/// Canonical production constructor. This is the only path that authorizes a
/// RelayTriad HeldOut campaign and therefore requires exact 256/128/64 counts.
pub(super) fn seal_relay_triad_heldout(
    legacy_candidate: &CandidateSubjectBindingV1,
    development: &RelayDevelopmentArtifact,
    comparator: &RelayComparatorFreezeArtifact,
) -> Result<RelayHeldOutAuthorization, RelayHeldOutSealError> {
    revalidate_development(development, true)?;
    revalidate_comparator(development, comparator, true)?;

    let contract = RelayTriadFepTargetContract::new(&development.snapshot)
        .map_err(RelayHeldOutSealError::TargetContract)?;
    let scoped_candidate = RelayScopedCandidateBinding::build(
        legacy_candidate,
        &development.snapshot,
        &contract,
    )?;

    let profiles = relay_scheduled_profiles(CorpusPartition::HeldOutEvaluation);
    if profiles.len() != usize::from(RELAY_HELD_OUT_WORLDS) {
        return Err(RelayHeldOutSealError::HeldOutScheduleMismatch);
    }
    let (profile_root, world_root, action_root, target_action_root, public_actions, target_actions) =
        freeze_heldout_schedule(&profiles, &contract)?;

    let selected_comparator = comparator
        .selection
        .selected
        .ok_or(RelayHeldOutSealError::ComparatorNotSelected)?;
    let legacy_candidate_digest = legacy_candidate
        .replay_digest()
        .map_err(RelayHeldOutSealError::InvalidLegacyCandidate)?;

    let mut receipt = RelayHeldOutSealReceipt {
        revision: RELAY_HELDOUT_SEAL_REVISION,
        analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
        legacy_candidate_digest,
        scoped_candidate_digest: scoped_candidate.replay_digest(),
        learned_snapshot_replay_digest: development.snapshot.replay_digest(),
        target_contract_digest: contract.replay_digest(),
        target_scope: EurekaTargetScope::ProductionFepComponentSnapshot,
        family_id: RELAY_TRIAD_FAMILY_ID,
        development_receipt_digest: development.receipt.replay_digest,
        comparator_freeze_receipt_digest: comparator.receipt.replay_digest,
        comparator_selection_replay_digest: comparator.selection.replay_digest(),
        selected_comparator,
        schedule_revision: RELAY_TRIAD_SCHEDULE_REVISION,
        heldout_profile_root: profile_root,
        heldout_world_root: world_root,
        heldout_public_action_root: action_root,
        heldout_target_action_root: target_action_root,
        heldout_world_count: profiles.len() as u32,
        outcome_count_at_seal: RELAY_HELDOUT_OUTCOMES_AT_SEAL,
        target_prediction_count_at_seal: RELAY_HELDOUT_TARGET_PREDICTIONS_AT_SEAL,
        replay_digest: 0,
    };
    receipt.replay_digest = seal_receipt_digest(&receipt);

    Ok(RelayHeldOutAuthorization {
        receipt,
        snapshot: development.snapshot.clone(),
        contract,
        scoped_candidate,
        profiles,
        public_actions,
        target_actions,
    })
}

fn revalidate_development(
    development: &RelayDevelopmentArtifact,
    require_canonical_count: bool,
) -> Result<(), RelayHeldOutSealError> {
    let receipt = &development.receipt;
    if receipt.runner_revision != RELAY_DEVELOPMENT_RUNNER_REVISION
        || receipt.family_id != RELAY_TRIAD_FAMILY_ID
        || receipt.schedule_revision != RELAY_TRIAD_SCHEDULE_REVISION
        || receipt.target_adapter_revision != RELAY_TRIAD_TARGET_ADAPTER_REVISION
        || receipt.target_scope != EurekaTargetScope::ProductionFepComponentSnapshot
        || receipt.partition != CorpusPartition::Development
        || receipt.learned_snapshot_replay_digest != development.snapshot.replay_digest()
        || receipt.target_observation_dim != development.snapshot.observation_dim()
        || receipt.target_action_count != development.snapshot.action_count()
    {
        return Err(RelayHeldOutSealError::DevelopmentIdentityMismatch);
    }
    if receipt.world_count as usize != development.records.len()
        || receipt.prediction_count != receipt.world_count
        || receipt.learning_count != receipt.world_count
        || receipt.action_histogram.iter().sum::<u32>() != receipt.world_count
        || (require_canonical_count && receipt.world_count != 256)
    {
        return Err(RelayHeldOutSealError::DevelopmentCountMismatch);
    }
    if development
        .records
        .iter()
        .any(|record| record.partition() != CorpusPartition::Development)
    {
        return Err(RelayHeldOutSealError::DevelopmentRecordMismatch);
    }

    let profiles = relay_scheduled_profiles(CorpusPartition::Development);
    if require_canonical_count && profiles.len() != development.records.len() {
        return Err(RelayHeldOutSealError::DevelopmentCountMismatch);
    }
    let profiles = &profiles[..development.records.len().min(profiles.len())];
    let mut profile_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut transition_bytes = Vec::with_capacity(profiles.len() * 8);
    for (profile, record) in profiles.iter().zip(&development.records) {
        if record.profile_digest() != profile.replay_digest() {
            return Err(RelayHeldOutSealError::DevelopmentRecordMismatch);
        }
        profile_bytes.extend_from_slice(&profile.replay_digest().to_le_bytes());
        transition_bytes.extend_from_slice(&record.transition_digest().to_le_bytes());
    }
    if domain_hash(b"eureka.002n.relay-development-profiles.v1\0", &profile_bytes)
        != receipt.scheduled_profile_root
    {
        return Err(RelayHeldOutSealError::DevelopmentProfileRootMismatch);
    }
    if domain_hash(
        b"eureka.002n.relay-development-transitions.v1\0",
        &transition_bytes,
    ) != receipt.realized_transition_root
    {
        return Err(RelayHeldOutSealError::DevelopmentTransitionRootMismatch);
    }
    if development_receipt_digest(receipt) != receipt.replay_digest {
        return Err(RelayHeldOutSealError::DevelopmentReceiptMismatch);
    }
    Ok(())
}

fn revalidate_comparator(
    development: &RelayDevelopmentArtifact,
    comparator: &RelayComparatorFreezeArtifact,
    require_canonical_count: bool,
) -> Result<(), RelayHeldOutSealError> {
    let receipt = &comparator.receipt;
    if receipt.runner_revision != RELAY_COMPARATOR_FREEZE_REVISION
        || receipt.family_id != RELAY_TRIAD_FAMILY_ID
        || receipt.schedule_revision != RELAY_TRIAD_SCHEDULE_REVISION
        || receipt.development_receipt_digest != development.receipt.replay_digest
        || receipt.learned_snapshot_replay_digest != development.snapshot.replay_digest()
    {
        return Err(RelayHeldOutSealError::ComparatorIdentityMismatch);
    }
    if receipt.target_prediction_count != RELAY_TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE {
        return Err(RelayHeldOutSealError::ComparatorTargetLeakage);
    }
    if receipt.calibration_world_count as usize != comparator.calibration_records.len()
        || (require_canonical_count && receipt.calibration_world_count != 128)
    {
        return Err(RelayHeldOutSealError::ComparatorCountMismatch);
    }
    if comparator
        .calibration_records
        .iter()
        .any(|record| record.partition() != CorpusPartition::Calibration)
    {
        return Err(RelayHeldOutSealError::CalibrationRecordMismatch);
    }

    let profiles = relay_scheduled_profiles(CorpusPartition::Calibration);
    if require_canonical_count && profiles.len() != comparator.calibration_records.len() {
        return Err(RelayHeldOutSealError::ComparatorCountMismatch);
    }
    let profiles = &profiles[..comparator.calibration_records.len().min(profiles.len())];
    let mut profile_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut transition_bytes = Vec::with_capacity(profiles.len() * 8);
    for (profile, record) in profiles.iter().zip(&comparator.calibration_records) {
        if record.profile_digest() != profile.replay_digest() {
            return Err(RelayHeldOutSealError::CalibrationRecordMismatch);
        }
        profile_bytes.extend_from_slice(&profile.replay_digest().to_le_bytes());
        transition_bytes.extend_from_slice(&record.transition_digest().to_le_bytes());
    }
    if domain_hash(b"eureka.002o.relay-calibration-profiles.v1\0", &profile_bytes)
        != receipt.calibration_profile_root
    {
        return Err(RelayHeldOutSealError::CalibrationProfileRootMismatch);
    }
    if domain_hash(
        b"eureka.002o.relay-calibration-transitions.v1\0",
        &transition_bytes,
    ) != receipt.calibration_transition_root
    {
        return Err(RelayHeldOutSealError::CalibrationTransitionRootMismatch);
    }

    let fit_digest = relay_corpus_digest_development(&development.records);
    if fit_digest != receipt.development_fit_corpus_digest
        || fit_digest != comparator.fitted.fit_corpus_digest()
        || fit_digest != comparator.selection.fit_corpus_digest
    {
        return Err(RelayHeldOutSealError::FitCorpusDigestMismatch);
    }
    let selection_digest = relay_corpus_digest_calibration(&comparator.calibration_records);
    if selection_digest != receipt.calibration_selection_corpus_digest
        || selection_digest != comparator.selection.selection_corpus_digest
    {
        return Err(RelayHeldOutSealError::SelectionCorpusDigestMismatch);
    }
    if comparator.selection.replay_digest() != receipt.comparator_selection_replay_digest
        || comparator.selection.status != receipt.selection_status
        || comparator.selection.selected != receipt.selected
        || comparator.selection.analysis_plan_digest != EUREKA_002_ANALYSIS_PLAN_V1.replay_digest()
    {
        return Err(RelayHeldOutSealError::SelectionIdentityMismatch);
    }
    if comparator.selection.status != PrimaryComparatorSelectionStatus::Selected
        || comparator.selection.selected.is_none()
    {
        return Err(RelayHeldOutSealError::ComparatorNotSelected);
    }
    if comparator_receipt_digest(receipt) != receipt.replay_digest {
        return Err(RelayHeldOutSealError::ComparatorIdentityMismatch);
    }
    Ok(())
}

fn freeze_heldout_schedule(
    profiles: &[RelayTriadProfile],
    contract: &RelayTriadFepTargetContract,
) -> Result<(u64, u64, u64, u64, Vec<PublicAction>, Vec<usize>), RelayHeldOutSealError> {
    let mut profile_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut world_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut action_bytes = Vec::new();
    let mut target_action_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut public_actions = Vec::with_capacity(profiles.len());
    let mut target_actions = Vec::with_capacity(profiles.len());

    for profile in profiles {
        if profile.partition != CorpusPartition::HeldOutEvaluation
            || profile.schedule_revision != RELAY_TRIAD_SCHEDULE_REVISION
        {
            return Err(RelayHeldOutSealError::HeldOutScheduleMismatch);
        }
        let profile_digest = profile.replay_digest();
        profile_bytes.extend_from_slice(&profile_digest.to_le_bytes());

        let mut evaluator = RelayTriadEvaluator::build(*profile);
        world_bytes.extend_from_slice(&evaluator.world_digest().to_le_bytes());
        let legal_actions = evaluator.runtime().legal_actions();
        let action = relay_scheduled_action(*profile, &legal_actions)
            .ok_or(RelayHeldOutSealError::HeldOutNoLegalAction)?;
        encode_action(&mut action_bytes, action);
        let target_action = contract
            .encode_action(action)
            .map_err(RelayHeldOutSealError::TargetContract)?;
        if target_action >= development_target_action_limit(contract) {
            return Err(RelayHeldOutSealError::HeldOutTargetActionMismatch);
        }
        target_action_bytes.extend_from_slice(&(target_action as u64).to_le_bytes());
        public_actions.push(action);
        target_actions.push(target_action);
    }

    Ok((
        domain_hash(b"eureka.002p.relay-heldout-profiles.v1\0", &profile_bytes),
        domain_hash(b"eureka.002p.relay-heldout-worlds.v1\0", &world_bytes),
        domain_hash(b"eureka.002p.relay-heldout-actions.v1\0", &action_bytes),
        domain_hash(
            b"eureka.002p.relay-heldout-target-actions.v1\0",
            &target_action_bytes,
        ),
        public_actions,
        target_actions,
    ))
}

fn development_target_action_limit(_contract: &RelayTriadFepTargetContract) -> usize {
    // RelayTriad's frozen adapter is exactly a 4-action canonical vocabulary.
    4
}

fn scoped_candidate_digest(binding: &RelayScopedCandidateBinding) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.002p.relay-scoped-candidate.v1\0");
    bytes.extend_from_slice(&binding.legacy_candidate_digest.to_le_bytes());
    bytes.push(match binding.scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    encode_str(&mut bytes, binding.family_id);
    bytes.extend_from_slice(&binding.snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&binding.target_contract_digest.to_le_bytes());
    encode_str(&mut bytes, binding.adapter_revision);
    fnv1a64(&bytes)
}

fn seal_receipt_digest(receipt: &RelayHeldOutSealReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.revision);
    bytes.extend_from_slice(&receipt.analysis_plan_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.legacy_candidate_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.scoped_candidate_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.learned_snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.target_contract_digest.to_le_bytes());
    bytes.push(match receipt.target_scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    encode_str(&mut bytes, receipt.family_id);
    bytes.extend_from_slice(&receipt.development_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_freeze_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_selection_replay_digest.to_le_bytes());
    encode_str(&mut bytes, receipt.selected_comparator.stable_id());
    encode_str(&mut bytes, receipt.schedule_revision);
    bytes.extend_from_slice(&receipt.heldout_profile_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.heldout_world_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.heldout_public_action_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.heldout_target_action_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.heldout_world_count.to_le_bytes());
    bytes.extend_from_slice(&receipt.outcome_count_at_seal.to_le_bytes());
    bytes.extend_from_slice(&receipt.target_prediction_count_at_seal.to_le_bytes());
    fnv1a64(&bytes)
}

fn development_receipt_digest(receipt: &RelayDevelopmentReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.runner_revision);
    encode_str(&mut bytes, receipt.family_id);
    encode_str(&mut bytes, receipt.schedule_revision);
    encode_str(&mut bytes, receipt.target_adapter_revision);
    bytes.push(match receipt.target_scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    bytes.push(match receipt.partition {
        CorpusPartition::Development => 1,
        CorpusPartition::Calibration => 2,
        CorpusPartition::HeldOutEvaluation => 3,
        CorpusPartition::ExternalReplication => 4,
    });
    bytes.extend_from_slice(&receipt.initial_snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.learned_snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.initial_target_contract_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.scheduled_profile_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.prospective_prediction_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.realized_transition_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.world_count.to_le_bytes());
    bytes.extend_from_slice(&receipt.prediction_count.to_le_bytes());
    bytes.extend_from_slice(&receipt.learning_count.to_le_bytes());
    bytes.extend_from_slice(&(receipt.target_observation_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&(receipt.target_action_count as u64).to_le_bytes());
    bytes.extend_from_slice(&(receipt.action_histogram.len() as u64).to_le_bytes());
    for count in &receipt.action_histogram {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    fnv1a64(&bytes)
}

fn comparator_receipt_digest(
    receipt: &super::relay_triad_comparator::RelayComparatorFreezeReceipt,
) -> u64 {
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

fn relay_corpus_digest_development(records: &[RelayDevelopmentTransitionRecord]) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.relay-fit-corpus.v1\0");
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    let mut ids: Vec<_> = records.iter().map(|record| record.transition_digest()).collect();
    ids.sort_unstable();
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.push(1);
    }
    fnv1a64(&bytes)
}

fn relay_corpus_digest_calibration(
    records: &[super::relay_triad_comparator::RelayCalibrationTransitionRecord],
) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.relay-calibration-selection-corpus.v1\0");
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    let mut ids: Vec<_> = records.iter().map(|record| record.transition_digest()).collect();
    ids.sort_unstable();
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.push(2);
    }
    fnv1a64(&bytes)
}

fn encode_action(bytes: &mut Vec<u8>, action: PublicAction) {
    match action {
        PublicAction::NoOp => bytes.push(1),
        PublicAction::Pulse { slot } => {
            bytes.push(2);
            bytes.push(slot);
        }
        PublicAction::Transfer { from, to, amount } => {
            bytes.push(3);
            bytes.push(from);
            bytes.push(to);
            bytes.extend_from_slice(&amount.to_le_bytes());
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
    use std::iter::repeat_n;

    fn sha(ch: char) -> String {
        repeat_n(ch, 64).collect()
    }

    fn legacy() -> CandidateSubjectBindingV1 {
        CandidateSubjectBindingV1 {
            source_commit_hex: repeat_n('a', 40).collect(),
            source_tree_sha256: sha('b'),
            candidate_config_sha256: sha('c'),
            target_adapter_revision: RELAY_TRIAD_TARGET_ADAPTER_REVISION.to_string(),
            flake_lock_sha256: sha('d'),
            toolchain_evidence_sha256: sha('e'),
            execution_environment_sha256: sha('f'),
            command_runner_sha256: sha('1'),
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
        }
    }

    #[test]
    fn heldout_schedule_freezes_64_worlds_without_execution() {
        let snapshot = {
            use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, FepPredictionSession};
            let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
                state_dim: 8,
                obs_dim: 4,
                num_actions: 4,
                enable_td_learning: true,
                ..Default::default()
            });
            FepPredictionSession::from_agent(&agent)
                .freeze_for_evaluation()
                .unwrap()
        };
        let contract = RelayTriadFepTargetContract::new(&snapshot).unwrap();
        let profiles = relay_scheduled_profiles(CorpusPartition::HeldOutEvaluation);
        let (_, _, _, _, actions, target_actions) = freeze_heldout_schedule(&profiles, &contract).unwrap();
        assert_eq!(profiles.len(), 64);
        assert_eq!(actions.len(), 64);
        assert_eq!(target_actions.len(), 64);
        assert!(target_actions.iter().all(|action| *action < 4));
    }

    #[test]
    fn scoped_candidate_rejects_resourceflow_adapter_revision() {
        let mut candidate = legacy();
        candidate.target_adapter_revision = super::super::target_contract::FEP_TARGET_ADAPTER_REVISION.to_string();
        let snapshot = {
            use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, FepPredictionSession};
            let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
                state_dim: 8,
                obs_dim: 4,
                num_actions: 4,
                enable_td_learning: true,
                ..Default::default()
            });
            FepPredictionSession::from_agent(&agent)
                .freeze_for_evaluation()
                .unwrap()
        };
        let contract = RelayTriadFepTargetContract::new(&snapshot).unwrap();
        assert_eq!(
            RelayScopedCandidateBinding::build(&candidate, &snapshot, &contract),
            Err(RelayHeldOutSealError::AdapterRevisionMismatch)
        );
    }

    #[test]
    fn scoped_candidate_identity_is_deterministic() {
        let snapshot = {
            use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, FepPredictionSession};
            let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
                state_dim: 8,
                obs_dim: 4,
                num_actions: 4,
                enable_td_learning: true,
                ..Default::default()
            });
            FepPredictionSession::from_agent(&agent)
                .freeze_for_evaluation()
                .unwrap()
        };
        let contract = RelayTriadFepTargetContract::new(&snapshot).unwrap();
        let a = RelayScopedCandidateBinding::build(&legacy(), &snapshot, &contract).unwrap();
        let b = RelayScopedCandidateBinding::build(&legacy(), &snapshot, &contract).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.replay_digest(), b.replay_digest());
    }
}
