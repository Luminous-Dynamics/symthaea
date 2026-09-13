// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Development-only production-FEP learning on RelayTriad.
//!
//! This module has no Calibration, HeldOut or External target runner. It learns
//! only from the independent RelayTriad Development schedule and freezes the
//! resulting detached production-FEP model for later tranches.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2234>

use symthaea::cognitive_loop::CognitiveLoopService;
use symthaea_fep::{FepEvaluationSnapshot, FepPredictionSessionError};

use super::action_execution::ActionExecutionStatus;
use super::hidden_world::{CorpusPartition, PublicAction, PublicObservation};
use super::relay_triad::{
    RELAY_DEVELOPMENT_WORLDS, RELAY_TRIAD_FAMILY_ID, RELAY_TRIAD_SCHEDULE_REVISION,
    RELAY_TRIAD_TARGET_ADAPTER_REVISION, RelayTriadEvaluator, RelayTriadFepTargetContract,
    RelayTriadProfile, RelayTriadTargetContractError, relay_scheduled_action,
    relay_scheduled_profiles,
};
use super::target_contract::EurekaTargetScope;

pub(super) const RELAY_DEVELOPMENT_RUNNER_REVISION: &str =
    "EUREKA.002N.RELAY_TRIAD_DEVELOPMENT.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayDevelopmentTransitionRecord {
    world_digest: u64,
    transition_digest: u64,
    partition: CorpusPartition,
    profile_digest: u64,
    pre_state: PublicObservation,
    action: PublicAction,
    post_state: PublicObservation,
}

impl RelayDevelopmentTransitionRecord {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RelayDevelopmentError {
    EmptyProfileSet,
    WrongPartition,
    WrongScheduleRevision,
    TargetContract(RelayTriadTargetContractError),
    Session(FepPredictionSessionError),
    NoScheduledAction,
    TargetActionMismatch,
    RealizedActionMismatch,
    MissingPostState,
    MissingTransitionIdentity,
    TransitionPreStateMismatch,
    UnexpectedMappedAction,
    SourceServiceChanged,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayDevelopmentReceipt {
    pub runner_revision: &'static str,
    pub family_id: &'static str,
    pub schedule_revision: &'static str,
    pub target_adapter_revision: &'static str,
    pub target_scope: EurekaTargetScope,
    pub partition: CorpusPartition,
    pub initial_snapshot_replay_digest: u64,
    pub learned_snapshot_replay_digest: u64,
    pub initial_target_contract_digest: u64,
    pub scheduled_profile_root: u64,
    pub prospective_prediction_root: u64,
    pub realized_transition_root: u64,
    pub world_count: u32,
    pub prediction_count: u32,
    pub learning_count: u32,
    pub target_observation_dim: usize,
    pub target_action_count: usize,
    pub action_histogram: Vec<u32>,
    pub replay_digest: u64,
}

#[derive(Debug, Clone)]
pub(super) struct RelayDevelopmentArtifact {
    pub snapshot: FepEvaluationSnapshot,
    pub receipt: RelayDevelopmentReceipt,
    pub records: Vec<RelayDevelopmentTransitionRecord>,
}

/// Canonical 256-world RelayTriad Development entry point.
pub(super) fn run_relay_triad_development(
    service: &CognitiveLoopService,
) -> Result<RelayDevelopmentArtifact, RelayDevelopmentError> {
    let profiles = relay_scheduled_profiles(CorpusPartition::Development);
    debug_assert_eq!(profiles.len(), usize::from(RELAY_DEVELOPMENT_WORLDS));
    run_relay_triad_development_profiles(service, &profiles)
}

/// Testable executor over a Development subset. The full profile set is
/// validated before the first detached-model update occurs.
pub(super) fn run_relay_triad_development_profiles(
    service: &CognitiveLoopService,
    profiles: &[RelayTriadProfile],
) -> Result<RelayDevelopmentArtifact, RelayDevelopmentError> {
    if profiles.is_empty() {
        return Err(RelayDevelopmentError::EmptyProfileSet);
    }
    for profile in profiles {
        if profile.partition != CorpusPartition::Development {
            return Err(RelayDevelopmentError::WrongPartition);
        }
        if profile.schedule_revision != RELAY_TRIAD_SCHEDULE_REVISION {
            return Err(RelayDevelopmentError::WrongScheduleRevision);
        }
    }

    let source_snapshot = service
        .fep_prediction_session()
        .freeze_for_evaluation()
        .map_err(RelayDevelopmentError::Session)?;
    let initial_snapshot_replay_digest = source_snapshot.replay_digest();
    let contract = RelayTriadFepTargetContract::new(&source_snapshot)
        .map_err(RelayDevelopmentError::TargetContract)?;

    let target_observation_dim = source_snapshot.observation_dim();
    let target_action_count = source_snapshot.action_count();
    let initial_target_contract_digest = contract.replay_digest();

    let mut session = service.fep_prediction_session();
    session.reset_transient_state_preserving_model();

    let mut records = Vec::with_capacity(profiles.len());
    let mut profile_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut prediction_bytes = Vec::new();
    let mut transition_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut action_histogram = vec![0_u32; target_action_count];
    let mut prediction_count = 0_u32;
    let mut learning_count = 0_u32;

    for profile in profiles {
        session.reset_transient_state_preserving_model();
        let profile_digest = profile.replay_digest();
        profile_bytes.extend_from_slice(&profile_digest.to_le_bytes());

        let mut evaluator = RelayTriadEvaluator::build(*profile);
        let pre_state = evaluator.runtime().observe();
        let legal_actions = evaluator.runtime().legal_actions();
        let public_action = relay_scheduled_action(*profile, &legal_actions)
            .ok_or(RelayDevelopmentError::NoScheduledAction)?;
        let target_observation = contract
            .encode_observation(&pre_state)
            .map_err(RelayDevelopmentError::TargetContract)?;
        let target_action = contract
            .encode_action(public_action)
            .map_err(RelayDevelopmentError::TargetContract)?;
        let Some(action_count) = action_histogram.get_mut(target_action) else {
            return Err(RelayDevelopmentError::UnexpectedMappedAction);
        };

        session
            .observe(&target_observation, 1.0, "eureka-relay-development")
            .map_err(RelayDevelopmentError::Session)?;

        // Commit the target prediction before evaluator execution reveals the
        // fresh post-state.
        let prediction = session
            .predict(target_action)
            .map_err(RelayDevelopmentError::Session)?;
        if prediction.action != target_action {
            return Err(RelayDevelopmentError::TargetActionMismatch);
        }
        encode_prediction(
            &mut prediction_bytes,
            profile_digest,
            &prediction,
        );
        prediction_count = prediction_count.saturating_add(1);

        let realization = evaluator.execute_qualified_action(public_action);
        if realization.status != ActionExecutionStatus::Applied
            || realization.realized != Some(public_action)
        {
            return Err(RelayDevelopmentError::RealizedActionMismatch);
        }
        if realization.pre_state != pre_state {
            return Err(RelayDevelopmentError::TransitionPreStateMismatch);
        }
        let post_state = realization
            .post_state
            .ok_or(RelayDevelopmentError::MissingPostState)?;
        let transition_digest = realization
            .transition_digest
            .ok_or(RelayDevelopmentError::MissingTransitionIdentity)?;
        transition_bytes.extend_from_slice(&transition_digest.to_le_bytes());

        let actual_observation = contract
            .encode_observation(&post_state)
            .map_err(RelayDevelopmentError::TargetContract)?;
        session
            .learn_from_actual(
                target_action,
                &actual_observation,
                1.0,
                "eureka-relay-development",
            )
            .map_err(RelayDevelopmentError::Session)?;
        learning_count = learning_count.saturating_add(1);
        *action_count = action_count.saturating_add(1);

        records.push(RelayDevelopmentTransitionRecord {
            world_digest: realization.world_digest,
            transition_digest,
            partition: CorpusPartition::Development,
            profile_digest,
            pre_state,
            action: public_action,
            post_state,
        });
    }

    let learned_snapshot = session
        .freeze_for_evaluation()
        .map_err(RelayDevelopmentError::Session)?;

    // The live service is immutable during the whole campaign. Recheck its
    // production-FEP identity so this is an executable invariant, not a comment.
    let source_after = service
        .fep_prediction_session()
        .freeze_for_evaluation()
        .map_err(RelayDevelopmentError::Session)?;
    if source_after.replay_digest() != initial_snapshot_replay_digest {
        return Err(RelayDevelopmentError::SourceServiceChanged);
    }

    let mut receipt = RelayDevelopmentReceipt {
        runner_revision: RELAY_DEVELOPMENT_RUNNER_REVISION,
        family_id: RELAY_TRIAD_FAMILY_ID,
        schedule_revision: RELAY_TRIAD_SCHEDULE_REVISION,
        target_adapter_revision: RELAY_TRIAD_TARGET_ADAPTER_REVISION,
        target_scope: EurekaTargetScope::ProductionFepComponentSnapshot,
        partition: CorpusPartition::Development,
        initial_snapshot_replay_digest,
        learned_snapshot_replay_digest: learned_snapshot.replay_digest(),
        initial_target_contract_digest,
        scheduled_profile_root: domain_hash(
            b"eureka.002n.relay-development-profiles.v1\0",
            &profile_bytes,
        ),
        prospective_prediction_root: domain_hash(
            b"eureka.002n.relay-development-predictions.v1\0",
            &prediction_bytes,
        ),
        realized_transition_root: domain_hash(
            b"eureka.002n.relay-development-transitions.v1\0",
            &transition_bytes,
        ),
        world_count: profiles.len() as u32,
        prediction_count,
        learning_count,
        target_observation_dim,
        target_action_count,
        action_histogram,
        replay_digest: 0,
    };
    receipt.replay_digest = receipt_digest(&receipt);

    Ok(RelayDevelopmentArtifact {
        snapshot: learned_snapshot,
        receipt,
        records,
    })
}

fn encode_prediction(
    bytes: &mut Vec<u8>,
    profile_digest: u64,
    prediction: &symthaea_fep::ActionOutcome,
) {
    bytes.extend_from_slice(&profile_digest.to_le_bytes());
    bytes.extend_from_slice(&(prediction.action as u64).to_le_bytes());
    encode_f64_slice(bytes, &prediction.predicted_next_state.mean);
    encode_f64_slice(bytes, &prediction.predicted_next_state.precision);
    encode_f64_slice(bytes, &prediction.expected_observation);
    bytes.extend_from_slice(&prediction.timestamp.to_le_bytes());
}

fn encode_f64_slice(bytes: &mut Vec<u8>, values: &[f64]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn receipt_digest(receipt: &RelayDevelopmentReceipt) -> u64 {
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
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    fn subset(partition: CorpusPartition, n: usize) -> Vec<RelayTriadProfile> {
        relay_scheduled_profiles(partition)
            .into_iter()
            .take(n)
            .collect()
    }

    #[test]
    fn canonical_entry_point_is_exact_256_development_worlds() {
        let profiles = relay_scheduled_profiles(CorpusPartition::Development);
        assert_eq!(profiles.len(), usize::from(RELAY_DEVELOPMENT_WORLDS));
        assert!(profiles.iter().all(|profile| {
            profile.partition == CorpusPartition::Development
                && profile.schedule_revision == RELAY_TRIAD_SCHEDULE_REVISION
        }));
    }

    #[test]
    fn development_subset_learns_detached_and_retains_public_records() {
        let service = service();
        let source_before = service
            .fep_prediction_session()
            .freeze_for_evaluation()
            .unwrap()
            .replay_digest();
        let artifact = run_relay_triad_development_profiles(
            &service,
            &subset(CorpusPartition::Development, 8),
        )
        .unwrap();
        let source_after = service
            .fep_prediction_session()
            .freeze_for_evaluation()
            .unwrap()
            .replay_digest();

        assert_eq!(source_before, source_after);
        assert_eq!(artifact.receipt.world_count, 8);
        assert_eq!(artifact.receipt.prediction_count, 8);
        assert_eq!(artifact.receipt.learning_count, 8);
        assert_eq!(artifact.receipt.action_histogram.iter().sum::<u32>(), 8);
        assert_eq!(artifact.records.len(), 8);
        assert!(artifact.records.iter().all(|record| {
            record.partition() == CorpusPartition::Development
                && record.pre_state().step == 0
                && record.post_state().step == 1
        }));
        assert_eq!(
            artifact.snapshot.replay_digest(),
            artifact.receipt.learned_snapshot_replay_digest
        );
    }

    #[test]
    fn wrong_partition_rejects_before_learning() {
        let service = service();
        let before = service
            .fep_prediction_session()
            .freeze_for_evaluation()
            .unwrap()
            .replay_digest();
        assert_eq!(
            run_relay_triad_development_profiles(
                &service,
                &subset(CorpusPartition::Calibration, 1),
            ),
            Err(RelayDevelopmentError::WrongPartition)
        );
        let after = service
            .fep_prediction_session()
            .freeze_for_evaluation()
            .unwrap()
            .replay_digest();
        assert_eq!(before, after);
    }

    #[test]
    fn tampered_schedule_revision_rejects_before_learning() {
        let service = service();
        let mut profiles = subset(CorpusPartition::Development, 1);
        profiles[0].schedule_revision = "not-relay-v1";
        assert_eq!(
            run_relay_triad_development_profiles(&service, &profiles),
            Err(RelayDevelopmentError::WrongScheduleRevision)
        );
    }

    #[test]
    fn same_source_and_schedule_replay_deterministically() {
        let profiles = subset(CorpusPartition::Development, 10);
        let a = run_relay_triad_development_profiles(&service(), &profiles).unwrap();
        let b = run_relay_triad_development_profiles(&service(), &profiles).unwrap();
        assert_eq!(a.receipt.scheduled_profile_root, b.receipt.scheduled_profile_root);
        assert_eq!(a.receipt.prospective_prediction_root, b.receipt.prospective_prediction_root);
        assert_eq!(a.receipt.realized_transition_root, b.receipt.realized_transition_root);
        assert_eq!(a.receipt.action_histogram, b.receipt.action_histogram);
        assert_eq!(a.snapshot.replay_digest(), b.snapshot.replay_digest());
        assert_eq!(a.receipt.replay_digest, b.receipt.replay_digest);
    }

    #[test]
    fn retained_records_bind_profile_and_transition_identity() {
        let profiles = subset(CorpusPartition::Development, 5);
        let artifact = run_relay_triad_development_profiles(&service(), &profiles).unwrap();
        for (profile, record) in profiles.iter().zip(&artifact.records) {
            assert_eq!(record.profile_digest(), profile.replay_digest());
            assert_ne!(record.world_digest(), 0);
            assert_ne!(record.transition_digest(), 0);
            assert!(matches!(
                record.action(),
                PublicAction::NoOp | PublicAction::Pulse { slot: 0..=2 }
            ));
        }
    }
}
