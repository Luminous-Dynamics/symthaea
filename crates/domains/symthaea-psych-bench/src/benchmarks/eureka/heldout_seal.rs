// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Zero-outcome precommitment seal for the first production-FEP HeldOut campaign.
//!
//! This module validates the complete pre-HeldOut evidence chain and freezes the
//! exact ResourceFlow HeldOut profiles/actions/world identities before any target
//! prediction or evaluator transition is executed.

use symthaea_fep::FepEvaluationSnapshot;

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::baselines::{PublicTransitionRecord, ShortcutBaselineKind, TransitionRecorder};
use super::campaign_manifest::{
    CALIBRATION_WORLDS_PER_FAMILY, CAMPAIGN_SCHEDULE_REVISION, CandidateSubjectBindingV1,
    CampaignScheduleError, DEVELOPMENT_WORLDS_PER_FAMILY, HELD_OUT_WORLDS_PER_FAMILY,
    ScheduledWorldProfile, scheduled_action, scheduled_profiles,
};
use super::fep_comparator_freeze::{
    ComparatorFreezeArtifact, RESOURCE_FLOW_COMPARATOR_FREEZE_REVISION,
    TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE,
};
use super::fep_development::{
    FEP_DEVELOPMENT_RUNNER_REVISION, FepDevelopmentArtifact, FepDevelopmentReceipt,
};
use super::hidden_world::{CorpusPartition, FixtureFamily, PublicAction};
use super::selection::{
    ComparatorSelectionCorpus, ComparatorSelectionError, PrimaryComparatorSelectionStatus,
    SelectionReadyFit, select_primary_comparator_v1,
};
use super::target_contract::{
    EurekaTargetScope, FEP_TARGET_ADAPTER_REVISION, FepTargetContract, FepTargetContractError,
};
use super::target_lineage::{
    TargetScopedCandidateBindingV2, TargetScopedCandidateError,
};

pub(super) const RESOURCE_FLOW_HELDOUT_SEAL_REVISION: &str =
    "EUREKA.002J.RESOURCE_FLOW_HELDOUT_SEAL.v1";

pub(super) const HELDOUT_OUTCOMES_AT_SEAL: u32 = 0;
pub(super) const HELDOUT_TARGET_PREDICTIONS_AT_SEAL: u32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HeldOutSealError {
    DevelopmentReceiptInvalid,
    DevelopmentWrongScope,
    DevelopmentWrongFamily,
    DevelopmentWrongPartition,
    DevelopmentNonCanonicalCount { expected: u32, actual: u32 },
    DevelopmentRecordMismatch,
    DevelopmentProfileRootMismatch,
    DevelopmentTransitionRootMismatch,
    LearnedSnapshotMismatch,
    TargetDimensionMismatch,
    ComparatorReceiptInvalid,
    ComparatorDevelopmentMismatch,
    ComparatorSnapshotMismatch,
    ComparatorTargetPredictionLeak,
    CalibrationNonCanonicalCount { expected: u32, actual: u32 },
    CalibrationRecordMismatch,
    CalibrationProfileRootMismatch,
    CalibrationTransitionRootMismatch,
    ComparatorFitDigestMismatch,
    ComparatorSelectionCorpusDigestMismatch,
    ComparatorSelectionMismatch,
    ComparatorReceiptSelectionMismatch,
    NoEligibleComparator,
    TargetContract(FepTargetContractError),
    ScopedCandidate(TargetScopedCandidateError),
    Selection(ComparatorSelectionError),
    Schedule(CampaignScheduleError),
}

/// Revalidated outcome-independent subject/comparator state.
#[derive(Debug, Clone)]
struct RevalidatedPreHeldOut {
    contract: FepTargetContract,
    scoped_candidate: TargetScopedCandidateBindingV2,
    selection_digest: u64,
    selected: Option<ShortcutBaselineKind>,
}

/// Evidence receipt for a sealed HeldOut authorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HeldOutSealReceipt {
    pub seal_revision: &'static str,
    pub analysis_plan_digest: u64,
    pub legacy_candidate_digest: u64,
    pub scoped_candidate_digest: u64,
    pub learned_snapshot_replay_digest: u64,
    pub target_contract_digest: u64,
    pub target_scope: EurekaTargetScope,
    pub family: FixtureFamily,
    pub development_receipt_digest: u64,
    pub comparator_freeze_receipt_digest: u64,
    pub comparator_selection_replay_digest: u64,
    pub selected_comparator: ShortcutBaselineKind,
    pub schedule_revision: &'static str,
    pub heldout_profile_root: u64,
    pub heldout_world_identity_root: u64,
    pub heldout_action_root: u64,
    pub heldout_world_count: u32,
    pub outcome_count_at_seal: u32,
    pub target_prediction_count_at_seal: u32,
    pub replay_digest: u64,
}

/// Exact frozen authorization consumed by a future, separately reviewed
/// HeldOut runner.  The learned snapshot is owned here so the execution subject
/// cannot drift between sealing and evaluation.
#[derive(Debug, Clone)]
pub(super) struct SealedHeldOutAuthorization {
    pub receipt: HeldOutSealReceipt,
    snapshot: FepEvaluationSnapshot,
    contract: FepTargetContract,
    scoped_candidate: TargetScopedCandidateBindingV2,
    profiles: Vec<ScheduledWorldProfile>,
    actions: Vec<PublicAction>,
}

impl SealedHeldOutAuthorization {
    pub(super) fn snapshot(&self) -> &FepEvaluationSnapshot {
        &self.snapshot
    }

    pub(super) fn contract(&self) -> &FepTargetContract {
        &self.contract
    }

    pub(super) fn scoped_candidate(&self) -> &TargetScopedCandidateBindingV2 {
        &self.scoped_candidate
    }

    pub(super) fn profiles(&self) -> &[ScheduledWorldProfile] {
        &self.profiles
    }

    pub(super) fn actions(&self) -> &[PublicAction] {
        &self.actions
    }
}

/// Canonical production constructor.  No HeldOut target prediction, observation
/// reveal, or transition execution occurs while this function runs.
pub(super) fn seal_resource_flow_heldout(
    legacy_candidate: &CandidateSubjectBindingV1,
    development: &FepDevelopmentArtifact,
    comparator: &ComparatorFreezeArtifact,
) -> Result<SealedHeldOutAuthorization, HeldOutSealError> {
    let revalidated = revalidate_preheldout(
        legacy_candidate,
        development,
        comparator,
        DEVELOPMENT_WORLDS_PER_FAMILY as u32,
        CALIBRATION_WORLDS_PER_FAMILY as u32,
    )?;
    let selected = revalidated.selected.ok_or(HeldOutSealError::NoEligibleComparator)?;
    let profiles = scheduled_profiles(
        CorpusPartition::HeldOutEvaluation,
        FixtureFamily::ResourceFlow,
    );
    if profiles.len() != usize::from(HELD_OUT_WORLDS_PER_FAMILY) {
        return Err(HeldOutSealError::DevelopmentNonCanonicalCount {
            expected: u32::from(HELD_OUT_WORLDS_PER_FAMILY),
            actual: profiles.len() as u32,
        });
    }
    build_authorization(
        development,
        comparator,
        revalidated,
        selected,
        profiles,
    )
}

/// Revalidate all pre-HeldOut artifacts from retained public records.  This
/// function intentionally does not require a selected comparator so the custody
/// theorem remains testable when calibration is legitimately inconclusive.
fn revalidate_preheldout(
    legacy_candidate: &CandidateSubjectBindingV1,
    development: &FepDevelopmentArtifact,
    comparator: &ComparatorFreezeArtifact,
    expected_development_worlds: u32,
    expected_calibration_worlds: u32,
) -> Result<RevalidatedPreHeldOut, HeldOutSealError> {
    validate_development(
        development,
        expected_development_worlds,
    )?;
    validate_comparator(
        development,
        comparator,
        expected_calibration_worlds,
    )?;

    let contract = FepTargetContract::new(
        &development.snapshot,
        FixtureFamily::ResourceFlow,
    )
    .map_err(HeldOutSealError::TargetContract)?;
    if contract.scope() != EurekaTargetScope::ProductionFepComponentSnapshot
        || contract.family() != FixtureFamily::ResourceFlow
    {
        return Err(HeldOutSealError::DevelopmentWrongScope);
    }
    if development.receipt.target_observation_dim != contract.observation_dim()
        || development.receipt.target_action_count != contract.action_count()
    {
        return Err(HeldOutSealError::TargetDimensionMismatch);
    }

    let scoped_candidate = TargetScopedCandidateBindingV2::from_production_fep(
        legacy_candidate,
        &contract,
    )
    .map_err(HeldOutSealError::ScopedCandidate)?;

    Ok(RevalidatedPreHeldOut {
        contract,
        scoped_candidate,
        selection_digest: comparator.selection.replay_digest(),
        selected: comparator.selection.selected,
    })
}

fn validate_development(
    development: &FepDevelopmentArtifact,
    expected_worlds: u32,
) -> Result<(), HeldOutSealError> {
    let receipt = &development.receipt;
    if receipt.replay_digest != development_receipt_digest(receipt) {
        return Err(HeldOutSealError::DevelopmentReceiptInvalid);
    }
    if receipt.runner_revision != FEP_DEVELOPMENT_RUNNER_REVISION
        || receipt.schedule_revision != CAMPAIGN_SCHEDULE_REVISION
        || receipt.target_adapter_revision != FEP_TARGET_ADAPTER_REVISION
    {
        return Err(HeldOutSealError::DevelopmentReceiptInvalid);
    }
    if receipt.target_scope != EurekaTargetScope::ProductionFepComponentSnapshot {
        return Err(HeldOutSealError::DevelopmentWrongScope);
    }
    if receipt.family != FixtureFamily::ResourceFlow {
        return Err(HeldOutSealError::DevelopmentWrongFamily);
    }
    if receipt.partition != CorpusPartition::Development {
        return Err(HeldOutSealError::DevelopmentWrongPartition);
    }
    if receipt.world_count != expected_worlds {
        return Err(HeldOutSealError::DevelopmentNonCanonicalCount {
            expected: expected_worlds,
            actual: receipt.world_count,
        });
    }
    if development.records.len() != expected_worlds as usize
        || receipt.prediction_count != expected_worlds
        || receipt.learning_count != expected_worlds
        || receipt.action_histogram.iter().copied().sum::<u32>() != expected_worlds
        || development.records.iter().any(|record| {
            record.partition() != CorpusPartition::Development
                || record.family() != FixtureFamily::ResourceFlow
        })
    {
        return Err(HeldOutSealError::DevelopmentRecordMismatch);
    }
    if development.snapshot.replay_digest() != receipt.learned_snapshot_replay_digest {
        return Err(HeldOutSealError::LearnedSnapshotMismatch);
    }

    let canonical = scheduled_profiles(CorpusPartition::Development, FixtureFamily::ResourceFlow);
    let expected_profiles = canonical
        .into_iter()
        .take(expected_worlds as usize)
        .collect::<Vec<_>>();
    let profile_root = scheduled_profile_root(
        b"eureka.002h.scheduled-profiles.v1\0",
        &expected_profiles,
    );
    if profile_root != receipt.scheduled_profile_root {
        return Err(HeldOutSealError::DevelopmentProfileRootMismatch);
    }
    let transition_root = transition_root(
        b"eureka.002h.realized-transitions.v1\0",
        &development.records,
    );
    if transition_root != receipt.realized_transition_root {
        return Err(HeldOutSealError::DevelopmentTransitionRootMismatch);
    }
    Ok(())
}

fn validate_comparator(
    development: &FepDevelopmentArtifact,
    comparator: &ComparatorFreezeArtifact,
    expected_worlds: u32,
) -> Result<(), HeldOutSealError> {
    let receipt = &comparator.receipt;
    if receipt.replay_digest != comparator_receipt_digest(receipt)
        || receipt.runner_revision != RESOURCE_FLOW_COMPARATOR_FREEZE_REVISION
        || receipt.schedule_revision != CAMPAIGN_SCHEDULE_REVISION
    {
        return Err(HeldOutSealError::ComparatorReceiptInvalid);
    }
    if receipt.development_receipt_digest != development.receipt.replay_digest {
        return Err(HeldOutSealError::ComparatorDevelopmentMismatch);
    }
    if receipt.learned_snapshot_replay_digest != development.snapshot.replay_digest() {
        return Err(HeldOutSealError::ComparatorSnapshotMismatch);
    }
    if receipt.target_prediction_count != TARGET_PREDICTIONS_DURING_COMPARATOR_FREEZE
        || receipt.target_prediction_count != 0
    {
        return Err(HeldOutSealError::ComparatorTargetPredictionLeak);
    }
    if receipt.calibration_world_count != expected_worlds {
        return Err(HeldOutSealError::CalibrationNonCanonicalCount {
            expected: expected_worlds,
            actual: receipt.calibration_world_count,
        });
    }
    if comparator.calibration_records.len() != expected_worlds as usize
        || comparator.calibration_records.iter().any(|record| {
            record.partition() != CorpusPartition::Calibration
                || record.family() != FixtureFamily::ResourceFlow
        })
    {
        return Err(HeldOutSealError::CalibrationRecordMismatch);
    }

    let canonical = scheduled_profiles(CorpusPartition::Calibration, FixtureFamily::ResourceFlow);
    let expected_profiles = canonical
        .into_iter()
        .take(expected_worlds as usize)
        .collect::<Vec<_>>();
    let profile_root = scheduled_profile_root(
        b"eureka.002i.calibration-profiles.v1\0",
        &expected_profiles,
    );
    if profile_root != receipt.calibration_profile_root {
        return Err(HeldOutSealError::CalibrationProfileRootMismatch);
    }
    let realized_root = transition_root(
        b"eureka.002i.calibration-transitions.v1\0",
        &comparator.calibration_records,
    );
    if realized_root != receipt.calibration_transition_root {
        return Err(HeldOutSealError::CalibrationTransitionRootMismatch);
    }

    let fit = SelectionReadyFit::freeze(development.records.clone())
        .map_err(HeldOutSealError::Selection)?;
    if fit.fit_corpus_digest() != receipt.development_fit_corpus_digest {
        return Err(HeldOutSealError::ComparatorFitDigestMismatch);
    }
    let selection_corpus = ComparatorSelectionCorpus::freeze(
        comparator.calibration_records.clone(),
    )
    .map_err(HeldOutSealError::Selection)?;
    if selection_corpus.digest() != receipt.calibration_selection_corpus_digest {
        return Err(HeldOutSealError::ComparatorSelectionCorpusDigestMismatch);
    }
    let recomputed = select_primary_comparator_v1(&fit, &selection_corpus)
        .map_err(HeldOutSealError::Selection)?;
    if recomputed != comparator.selection
        || recomputed.replay_digest() != receipt.comparator_selection_replay_digest
    {
        return Err(HeldOutSealError::ComparatorSelectionMismatch);
    }
    if receipt.selection_status != comparator.selection.status
        || receipt.selected != comparator.selection.selected
    {
        return Err(HeldOutSealError::ComparatorReceiptSelectionMismatch);
    }
    Ok(())
}

fn build_authorization(
    development: &FepDevelopmentArtifact,
    comparator: &ComparatorFreezeArtifact,
    revalidated: RevalidatedPreHeldOut,
    selected: ShortcutBaselineKind,
    profiles: Vec<ScheduledWorldProfile>,
) -> Result<SealedHeldOutAuthorization, HeldOutSealError> {
    if comparator.selection.status != PrimaryComparatorSelectionStatus::Selected
        || comparator.selection.selected != Some(selected)
    {
        return Err(HeldOutSealError::NoEligibleComparator);
    }

    let mut profile_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut world_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut action_bytes = Vec::new();
    let mut actions = Vec::with_capacity(profiles.len());

    for profile in &profiles {
        if profile.partition != CorpusPartition::HeldOutEvaluation
            || profile.family != FixtureFamily::ResourceFlow
            || profile.world.partition != CorpusPartition::HeldOutEvaluation
            || profile.world.family != FixtureFamily::ResourceFlow
        {
            return Err(HeldOutSealError::DevelopmentWrongPartition);
        }
        let profile_digest = profile.replay_digest();
        profile_bytes.extend_from_slice(&profile_digest.to_le_bytes());
        let mut recorder = TransitionRecorder::build(profile.world);
        world_bytes.extend_from_slice(&recorder.world_digest().to_le_bytes());
        let legal_actions = recorder.runtime().legal_actions();
        let action = scheduled_action(*profile, &legal_actions)
            .map_err(HeldOutSealError::Schedule)?;
        encode_scheduled_action(&mut action_bytes, profile_digest, action);
        actions.push(action);
    }

    let heldout_profile_root = domain_hash(
        b"eureka.002j.heldout-profiles.v1\0",
        &profile_bytes,
    );
    let heldout_world_identity_root = domain_hash(
        b"eureka.002j.heldout-world-identities.v1\0",
        &world_bytes,
    );
    let heldout_action_root = domain_hash(
        b"eureka.002j.heldout-actions.v1\0",
        &action_bytes,
    );

    let mut receipt = HeldOutSealReceipt {
        seal_revision: RESOURCE_FLOW_HELDOUT_SEAL_REVISION,
        analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
        legacy_candidate_digest: revalidated.scoped_candidate.legacy_candidate_digest(),
        scoped_candidate_digest: revalidated.scoped_candidate.replay_digest(),
        learned_snapshot_replay_digest: development.snapshot.replay_digest(),
        target_contract_digest: revalidated.contract.replay_digest(),
        target_scope: revalidated.contract.scope(),
        family: revalidated.contract.family(),
        development_receipt_digest: development.receipt.replay_digest,
        comparator_freeze_receipt_digest: comparator.receipt.replay_digest,
        comparator_selection_replay_digest: revalidated.selection_digest,
        selected_comparator: selected,
        schedule_revision: CAMPAIGN_SCHEDULE_REVISION,
        heldout_profile_root,
        heldout_world_identity_root,
        heldout_action_root,
        heldout_world_count: profiles.len() as u32,
        outcome_count_at_seal: HELDOUT_OUTCOMES_AT_SEAL,
        target_prediction_count_at_seal: HELDOUT_TARGET_PREDICTIONS_AT_SEAL,
        replay_digest: 0,
    };
    receipt.replay_digest = seal_receipt_digest(&receipt);

    Ok(SealedHeldOutAuthorization {
        receipt,
        snapshot: development.snapshot.clone(),
        contract: revalidated.contract,
        scoped_candidate: revalidated.scoped_candidate,
        profiles,
        actions,
    })
}

fn scheduled_profile_root(domain: &[u8], profiles: &[ScheduledWorldProfile]) -> u64 {
    let mut bytes = Vec::with_capacity(profiles.len() * 8);
    for profile in profiles {
        bytes.extend_from_slice(&profile.replay_digest().to_le_bytes());
    }
    domain_hash(domain, &bytes)
}

fn transition_root(domain: &[u8], records: &[PublicTransitionRecord]) -> u64 {
    let mut bytes = Vec::with_capacity(records.len() * 8);
    for record in records {
        bytes.extend_from_slice(&record.transition_digest().to_le_bytes());
    }
    domain_hash(domain, &bytes)
}

fn encode_scheduled_action(bytes: &mut Vec<u8>, profile_digest: u64, action: PublicAction) {
    bytes.extend_from_slice(&profile_digest.to_le_bytes());
    match action {
        PublicAction::NoOp => bytes.push(0),
        PublicAction::Pulse { slot } => {
            bytes.push(1);
            bytes.push(slot);
        }
        PublicAction::Transfer { from, to, amount } => {
            bytes.push(2);
            bytes.push(from);
            bytes.push(to);
            bytes.extend_from_slice(&amount.to_le_bytes());
        }
    }
}

fn development_receipt_digest(receipt: &FepDevelopmentReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.runner_revision);
    encode_str(&mut bytes, receipt.schedule_revision);
    encode_str(&mut bytes, receipt.target_adapter_revision);
    bytes.push(scope_tag(receipt.target_scope));
    bytes.push(family_tag(receipt.family));
    bytes.push(partition_tag(receipt.partition));
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
    receipt: &super::fep_comparator_freeze::ComparatorFreezeReceipt,
) -> u64 {
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

fn seal_receipt_digest(receipt: &HeldOutSealReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.seal_revision);
    bytes.extend_from_slice(&receipt.analysis_plan_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.legacy_candidate_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.scoped_candidate_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.learned_snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.target_contract_digest.to_le_bytes());
    bytes.push(scope_tag(receipt.target_scope));
    bytes.push(family_tag(receipt.family));
    bytes.extend_from_slice(&receipt.development_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_freeze_receipt_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.comparator_selection_replay_digest.to_le_bytes());
    encode_str(&mut bytes, receipt.selected_comparator.stable_id());
    encode_str(&mut bytes, receipt.schedule_revision);
    bytes.extend_from_slice(&receipt.heldout_profile_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.heldout_world_identity_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.heldout_action_root.to_le_bytes());
    bytes.extend_from_slice(&receipt.heldout_world_count.to_le_bytes());
    bytes.extend_from_slice(&receipt.outcome_count_at_seal.to_le_bytes());
    bytes.extend_from_slice(&receipt.target_prediction_count_at_seal.to_le_bytes());
    fnv1a64(&bytes)
}

fn scope_tag(scope: EurekaTargetScope) -> u8 {
    match scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    }
}

fn family_tag(family: FixtureFamily) -> u8 {
    match family {
        FixtureFamily::CausalBits => 1,
        FixtureFamily::ResourceFlow => 2,
    }
}

fn partition_tag(partition: CorpusPartition) -> u8 {
    match partition {
        CorpusPartition::Development => 1,
        CorpusPartition::Calibration => 2,
        CorpusPartition::HeldOutEvaluation => 3,
        CorpusPartition::ExternalReplication => 4,
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
    use crate::benchmarks::eureka::fep_comparator_freeze::freeze_resource_flow_comparator_profiles;
    use crate::benchmarks::eureka::fep_development::run_resource_flow_development_profiles;
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn legacy_candidate() -> CandidateSubjectBindingV1 {
        CandidateSubjectBindingV1 {
            source_commit_hex: std::iter::repeat_n('a', 40).collect(),
            source_tree_sha256: sha('b'),
            candidate_config_sha256: sha('c'),
            target_adapter_revision: FEP_TARGET_ADAPTER_REVISION.to_string(),
            flake_lock_sha256: sha('d'),
            toolchain_evidence_sha256: sha('e'),
            execution_environment_sha256: sha('f'),
            command_runner_sha256: sha('1'),
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
        }
    }

    fn service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    fn artifacts(
        development_worlds: usize,
        calibration_worlds: usize,
    ) -> (FepDevelopmentArtifact, ComparatorFreezeArtifact) {
        let development_profiles =
            scheduled_profiles(CorpusPartition::Development, FixtureFamily::ResourceFlow);
        let development = run_resource_flow_development_profiles(
            &service(),
            &development_profiles[..development_worlds],
        )
        .unwrap();
        let calibration_profiles =
            scheduled_profiles(CorpusPartition::Calibration, FixtureFamily::ResourceFlow);
        let comparator = freeze_resource_flow_comparator_profiles(
            &development,
            &calibration_profiles[..calibration_worlds],
        )
        .unwrap();
        (development, comparator)
    }

    #[test]
    fn retained_artifacts_revalidate_without_trusting_receipts() {
        let (development, comparator) = artifacts(12, 12);
        let result = revalidate_preheldout(
            &legacy_candidate(),
            &development,
            &comparator,
            12,
            12,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn comparator_target_prediction_leak_fails_closed() {
        let (development, mut comparator) = artifacts(8, 8);
        comparator.receipt.target_prediction_count = 1;
        assert_eq!(
            revalidate_preheldout(
                &legacy_candidate(),
                &development,
                &comparator,
                8,
                8,
            )
            .unwrap_err(),
            HeldOutSealError::ComparatorReceiptInvalid
        );
    }

    #[test]
    fn development_transition_tamper_fails_closed() {
        let (mut development, comparator) = artifacts(8, 8);
        development.records.swap(0, 1);
        assert_eq!(
            revalidate_preheldout(
                &legacy_candidate(),
                &development,
                &comparator,
                8,
                8,
            )
            .unwrap_err(),
            HeldOutSealError::DevelopmentTransitionRootMismatch
        );
    }

    #[test]
    fn heldout_schedule_freezes_profiles_worlds_and_actions_without_outcomes() {
        let profiles = scheduled_profiles(
            CorpusPartition::HeldOutEvaluation,
            FixtureFamily::ResourceFlow,
        );
        let subset = profiles[..7].to_vec();
        let (development, comparator) = artifacts(10, 10);
        let mut revalidated = revalidate_preheldout(
            &legacy_candidate(),
            &development,
            &comparator,
            10,
            10,
        )
        .unwrap();
        // This test exercises seal identity independent of whether a tiny
        // calibration subset happens to produce an eligible comparator.
        revalidated.selected = Some(ShortcutBaselineKind::NearestTransition);
        let auth = build_authorization(
            &development,
            &comparator,
            revalidated,
            ShortcutBaselineKind::NearestTransition,
            subset,
        );
        if comparator.selection.status == PrimaryComparatorSelectionStatus::Selected
            && comparator.selection.selected == Some(ShortcutBaselineKind::NearestTransition)
        {
            let auth = auth.unwrap();
            assert_eq!(auth.receipt.heldout_world_count, 7);
            assert_eq!(auth.receipt.outcome_count_at_seal, 0);
            assert_eq!(auth.receipt.target_prediction_count_at_seal, 0);
            assert_eq!(auth.profiles().len(), 7);
            assert_eq!(auth.actions().len(), 7);
        } else {
            assert_eq!(auth.unwrap_err(), HeldOutSealError::NoEligibleComparator);
        }
    }

    #[test]
    fn same_heldout_schedule_has_deterministic_precommitment_roots() {
        let profiles = scheduled_profiles(
            CorpusPartition::HeldOutEvaluation,
            FixtureFamily::ResourceFlow,
        );
        let mut profile_a = Vec::new();
        let mut world_a = Vec::new();
        let mut action_a = Vec::new();
        let mut profile_b = Vec::new();
        let mut world_b = Vec::new();
        let mut action_b = Vec::new();

        for (profile_bytes, world_bytes, action_bytes) in [
            (&mut profile_a, &mut world_a, &mut action_a),
            (&mut profile_b, &mut world_b, &mut action_b),
        ] {
            for profile in profiles.iter().take(9) {
                let pd = profile.replay_digest();
                profile_bytes.extend_from_slice(&pd.to_le_bytes());
                let mut recorder = TransitionRecorder::build(profile.world);
                world_bytes.extend_from_slice(&recorder.world_digest().to_le_bytes());
                let legal = recorder.runtime().legal_actions();
                let action = scheduled_action(*profile, &legal).unwrap();
                encode_scheduled_action(action_bytes, pd, action);
            }
        }
        assert_eq!(profile_a, profile_b);
        assert_eq!(world_a, world_b);
        assert_eq!(action_a, action_b);
    }

    #[test]
    fn canonical_heldout_schedule_is_exact_64_resource_flow_worlds() {
        let profiles = scheduled_profiles(
            CorpusPartition::HeldOutEvaluation,
            FixtureFamily::ResourceFlow,
        );
        assert_eq!(profiles.len(), usize::from(HELD_OUT_WORLDS_PER_FAMILY));
        assert!(profiles.iter().all(|profile| {
            profile.partition == CorpusPartition::HeldOutEvaluation
                && profile.family == FixtureFamily::ResourceFlow
        }));
    }
}
