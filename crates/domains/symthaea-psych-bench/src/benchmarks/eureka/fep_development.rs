// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Development-only learning campaign for the production FEP component target.
//!
//! This module intentionally has no Calibration/HeldOut/External target runner.
//! It may learn only from the frozen Development ResourceFlow schedule and emits
//! a frozen learned FEP snapshot for later, separately reviewed tranches.

use symthaea::cognitive_loop::CognitiveLoopService;
use symthaea_fep::{FepEvaluationSnapshot, FepPredictionSessionError};

use super::baselines::{PublicTransitionRecord, TransitionRecordError, TransitionRecorder};
use super::campaign_manifest::{
    CAMPAIGN_SCHEDULE_REVISION, CampaignScheduleError, ScheduledWorldProfile, scheduled_action,
    scheduled_profiles,
};
use super::hidden_world::{CorpusPartition, FixtureFamily};
use super::target_contract::{
    EurekaTargetScope, FEP_TARGET_ADAPTER_REVISION, FepTargetContract, FepTargetContractError,
};

pub(super) const FEP_DEVELOPMENT_RUNNER_REVISION: &str = "EUREKA.002H.FEP_DEVELOPMENT.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FepDevelopmentError {
    EmptyProfileSet,
    WrongPartition,
    WrongFamily,
    TargetContract(FepTargetContractError),
    Schedule(CampaignScheduleError),
    Transition(TransitionRecordError),
    Session(FepPredictionSessionError),
    TargetActionMismatch,
    RealizedActionMismatch,
    TransitionPreStateMismatch,
    UnexpectedMappedAction,
    SourceServiceChanged,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FepDevelopmentReceipt {
    pub runner_revision: &'static str,
    pub schedule_revision: &'static str,
    pub target_adapter_revision: &'static str,
    pub target_scope: EurekaTargetScope,
    pub family: FixtureFamily,
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
pub(super) struct FepDevelopmentArtifact {
    pub snapshot: FepEvaluationSnapshot,
    pub receipt: FepDevelopmentReceipt,
    /// Exact public Development transitions consumed by target learning. These
    /// are also the only records later baseline fitting is allowed to consume.
    pub records: Vec<PublicTransitionRecord>,
}

/// Run the exact canonical 256-world ResourceFlow Development schedule against
/// a detached clone of the FEP agent currently owned by the live service.
pub(super) fn run_resource_flow_development(
    service: &CognitiveLoopService,
) -> Result<FepDevelopmentArtifact, FepDevelopmentError> {
    let profiles = scheduled_profiles(CorpusPartition::Development, FixtureFamily::ResourceFlow);
    run_resource_flow_development_profiles(service, &profiles)
}

/// Internal/testable executor over a preselected Development subset. All
/// profiles are validated before any target learning occurs.
pub(super) fn run_resource_flow_development_profiles(
    service: &CognitiveLoopService,
    profiles: &[ScheduledWorldProfile],
) -> Result<FepDevelopmentArtifact, FepDevelopmentError> {
    if profiles.is_empty() {
        return Err(FepDevelopmentError::EmptyProfileSet);
    }
    for profile in profiles {
        if profile.partition != CorpusPartition::Development
            || profile.world.partition != CorpusPartition::Development
        {
            return Err(FepDevelopmentError::WrongPartition);
        }
        if profile.family != FixtureFamily::ResourceFlow
            || profile.world.family != FixtureFamily::ResourceFlow
        {
            return Err(FepDevelopmentError::WrongFamily);
        }
    }

    let source_snapshot = service
        .fep_prediction_session()
        .freeze_for_evaluation()
        .map_err(FepDevelopmentError::Session)?;
    let initial_snapshot_replay_digest = source_snapshot.replay_digest();
    let contract = FepTargetContract::new(&source_snapshot, FixtureFamily::ResourceFlow)
        .map_err(FepDevelopmentError::TargetContract)?;

    let mut session = service.fep_prediction_session();
    session.reset_transient_state_preserving_model();

    let mut records = Vec::with_capacity(profiles.len());
    let mut scheduled_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut prediction_bytes = Vec::new();
    let mut transition_bytes = Vec::with_capacity(profiles.len() * 8);
    let mut action_histogram = vec![0_u32; contract.action_count()];
    let mut prediction_count = 0_u32;
    let mut learning_count = 0_u32;

    for profile in profiles {
        // Every synthetic world is an independent episode. Preserve learned
        // model parameters but clear transient belief/action/eligibility state.
        session.reset_transient_state_preserving_model();

        scheduled_bytes.extend_from_slice(&profile.replay_digest().to_le_bytes());
        let mut recorder = TransitionRecorder::build(profile.world);
        let pre_state = recorder.runtime().observe();
        let legal_actions = recorder.runtime().legal_actions();
        let public_action = scheduled_action(*profile, &legal_actions)
            .map_err(FepDevelopmentError::Schedule)?;
        let target_observation = contract
            .encode_observation(&pre_state)
            .map_err(FepDevelopmentError::TargetContract)?;
        let target_action = contract
            .encode_action(public_action)
            .map_err(FepDevelopmentError::TargetContract)?;
        let Some(action_count) = action_histogram.get_mut(target_action) else {
            return Err(FepDevelopmentError::UnexpectedMappedAction);
        };

        session
            .observe(&target_observation, 1.0, "eureka-development")
            .map_err(FepDevelopmentError::Session)?;

        // Prospective prediction is created before the evaluator executes and
        // reveals the fresh transition outcome.
        let prediction = session
            .predict(target_action)
            .map_err(FepDevelopmentError::Session)?;
        if prediction.action != target_action {
            return Err(FepDevelopmentError::TargetActionMismatch);
        }
        prediction_count = prediction_count.saturating_add(1);
        encode_prediction_receipt(
            &mut prediction_bytes,
            profile.replay_digest(),
            &prediction,
        );

        let (realization, record) = recorder
            .execute_and_record(public_action)
            .map_err(FepDevelopmentError::Transition)?;
        if realization.realized != Some(public_action) || record.action() != public_action {
            return Err(FepDevelopmentError::RealizedActionMismatch);
        }
        if record.pre_state() != &pre_state {
            return Err(FepDevelopmentError::TransitionPreStateMismatch);
        }
        transition_bytes.extend_from_slice(&record.transition_digest().to_le_bytes());

        let actual_observation = contract
            .encode_observation(record.post_state())
            .map_err(FepDevelopmentError::TargetContract)?;
        session
            .learn_from_actual(
                target_action,
                &actual_observation,
                1.0,
                "eureka-development",
            )
            .map_err(FepDevelopmentError::Session)?;
        learning_count = learning_count.saturating_add(1);
        *action_count = action_count.saturating_add(1);
        records.push(record);
    }

    let learned_snapshot = session
        .freeze_for_evaluation()
        .map_err(FepDevelopmentError::Session)?;

    // The source service was borrowed immutably and all learning occurred in a
    // detached clone; verify its exact production-FEP replay identity anyway.
    let source_after = service
        .fep_prediction_session()
        .freeze_for_evaluation()
        .map_err(FepDevelopmentError::Session)?;
    if source_after.replay_digest() != initial_snapshot_replay_digest {
        return Err(FepDevelopmentError::SourceServiceChanged);
    }

    let mut receipt = FepDevelopmentReceipt {
        runner_revision: FEP_DEVELOPMENT_RUNNER_REVISION,
        schedule_revision: CAMPAIGN_SCHEDULE_REVISION,
        target_adapter_revision: FEP_TARGET_ADAPTER_REVISION,
        target_scope: contract.scope(),
        family: FixtureFamily::ResourceFlow,
        partition: CorpusPartition::Development,
        initial_snapshot_replay_digest,
        learned_snapshot_replay_digest: learned_snapshot.replay_digest(),
        initial_target_contract_digest: contract.replay_digest(),
        scheduled_profile_root: domain_hash(
            b"eureka.002h.scheduled-profiles.v1\0",
            &scheduled_bytes,
        ),
        prospective_prediction_root: domain_hash(
            b"eureka.002h.prospective-predictions.v1\0",
            &prediction_bytes,
        ),
        realized_transition_root: domain_hash(
            b"eureka.002h.realized-transitions.v1\0",
            &transition_bytes,
        ),
        world_count: profiles.len() as u32,
        prediction_count,
        learning_count,
        target_observation_dim: contract.observation_dim(),
        target_action_count: contract.action_count(),
        action_histogram,
        replay_digest: 0,
    };
    receipt.replay_digest = development_receipt_digest(&receipt);

    Ok(FepDevelopmentArtifact {
        snapshot: learned_snapshot,
        receipt,
        records,
    })
}

fn encode_prediction_receipt(
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

fn development_receipt_digest(receipt: &FepDevelopmentReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.runner_revision);
    encode_str(&mut bytes, receipt.schedule_revision);
    encode_str(&mut bytes, receipt.target_adapter_revision);
    bytes.push(match receipt.target_scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    bytes.push(match receipt.family {
        FixtureFamily::CausalBits => 1,
        FixtureFamily::ResourceFlow => 2,
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

    fn development_subset(n: usize) -> Vec<ScheduledWorldProfile> {
        scheduled_profiles(CorpusPartition::Development, FixtureFamily::ResourceFlow)
            .into_iter()
            .take(n)
            .collect()
    }

    #[test]
    fn canonical_entry_point_is_exact_256_world_resource_flow_schedule() {
        let profiles = scheduled_profiles(CorpusPartition::Development, FixtureFamily::ResourceFlow);
        assert_eq!(profiles.len(), 256);
        assert!(profiles.iter().all(|profile| {
            profile.partition == CorpusPartition::Development
                && profile.family == FixtureFamily::ResourceFlow
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
        let artifact = run_resource_flow_development_profiles(&service, &development_subset(6))
            .unwrap();
        let source_after = service
            .fep_prediction_session()
            .freeze_for_evaluation()
            .unwrap()
            .replay_digest();

        assert_eq!(source_before, source_after);
        assert_eq!(artifact.receipt.world_count, 6);
        assert_eq!(artifact.receipt.prediction_count, 6);
        assert_eq!(artifact.receipt.learning_count, 6);
        assert_eq!(artifact.records.len(), 6);
        assert_eq!(artifact.receipt.action_histogram.iter().sum::<u32>(), 6);
        assert_eq!(
            artifact.receipt.target_scope,
            EurekaTargetScope::ProductionFepComponentSnapshot
        );
        assert!(artifact.records.iter().all(|record| {
            record.partition() == CorpusPartition::Development
                && record.family() == FixtureFamily::ResourceFlow
        }));
        assert_eq!(
            artifact.snapshot.replay_digest(),
            artifact.receipt.learned_snapshot_replay_digest
        );
    }

    #[test]
    fn non_development_profiles_reject_before_training() {
        let service = service();
        let calibration = scheduled_profiles(
            CorpusPartition::Calibration,
            FixtureFamily::ResourceFlow,
        );
        assert!(matches!(
            run_resource_flow_development_profiles(&service, &calibration[..1]),
            Err(FepDevelopmentError::WrongPartition)
        ));
    }

    #[test]
    fn causal_bits_profiles_reject_before_training() {
        let service = service();
        let bits = scheduled_profiles(CorpusPartition::Development, FixtureFamily::CausalBits);
        assert!(matches!(
            run_resource_flow_development_profiles(&service, &bits[..1]),
            Err(FepDevelopmentError::WrongFamily)
        ));
    }

    #[test]
    fn same_source_and_schedule_replay_deterministically() {
        let a = run_resource_flow_development_profiles(&service(), &development_subset(5)).unwrap();
        let b = run_resource_flow_development_profiles(&service(), &development_subset(5)).unwrap();
        assert_eq!(a.receipt.scheduled_profile_root, b.receipt.scheduled_profile_root);
        assert_eq!(
            a.receipt.prospective_prediction_root,
            b.receipt.prospective_prediction_root
        );
        assert_eq!(
            a.receipt.realized_transition_root,
            b.receipt.realized_transition_root
        );
        assert_eq!(
            a.receipt.learned_snapshot_replay_digest,
            b.receipt.learned_snapshot_replay_digest
        );
        assert_eq!(a.receipt.replay_digest, b.receipt.replay_digest);
    }
}