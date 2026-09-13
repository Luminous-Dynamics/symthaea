// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RelayTriad — an isolated second EUREKA world family.
//!
//! This sidecar deliberately does not extend the frozen legacy `FixtureFamily`
//! enum or mutate EUREKA-002E V1 schedules. It has independent family, schedule,
//! world and target-contract identities while reusing only the public synthetic
//! observation/action vocabulary.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2214>

#![allow(dead_code)]

use symthaea_fep::FepEvaluationSnapshot;

use super::action_execution::ActionExecutionStatus;
use super::hidden_world::{
    CorpusPartition, InterventionReceipt, InterventionRequest, InterventionStatus, PublicAction,
    PublicObservation, PublicValue, StepReceipt, PUBLIC_SCHEMA_ID,
};
use super::target_contract::EurekaTargetScope;

pub(super) const RELAY_TRIAD_FAMILY_ID: &str = "EUREKA.WORLD.RELAY_TRIAD.v1";
pub(super) const RELAY_TRIAD_SCHEDULE_REVISION: &str =
    "EUREKA.002M.RELAY_TRIAD_SCHEDULE.v1";
pub(super) const RELAY_TRIAD_TARGET_ADAPTER_REVISION: &str =
    "EUREKA.TARGET.RELAY_TRIAD_FEP.v1";

pub(super) const RELAY_DEVELOPMENT_WORLDS: u16 = 256;
pub(super) const RELAY_CALIBRATION_WORLDS: u16 = 128;
pub(super) const RELAY_HELD_OUT_WORLDS: u16 = 64;
pub(super) const RELAY_EXTERNAL_WORLDS: u16 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct RelayTriadProfile {
    pub schedule_revision: &'static str,
    pub partition: CorpusPartition,
    pub index: u16,
    pub seed: u64,
    pub mechanism_variant: u8,
}

impl RelayTriadProfile {
    pub(super) fn replay_digest(self) -> u64 {
        let mut bytes = Vec::new();
        encode_str(&mut bytes, self.schedule_revision);
        encode_str(&mut bytes, RELAY_TRIAD_FAMILY_ID);
        bytes.push(partition_tag(self.partition));
        bytes.extend_from_slice(&self.index.to_le_bytes());
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        bytes.push(self.mechanism_variant);
        fnv1a64(&bytes)
    }
}

pub(super) fn relay_world_count(partition: CorpusPartition) -> u16 {
    match partition {
        CorpusPartition::Development => RELAY_DEVELOPMENT_WORLDS,
        CorpusPartition::Calibration => RELAY_CALIBRATION_WORLDS,
        CorpusPartition::HeldOutEvaluation => RELAY_HELD_OUT_WORLDS,
        CorpusPartition::ExternalReplication => RELAY_EXTERNAL_WORLDS,
    }
}

pub(super) fn relay_scheduled_profiles(partition: CorpusPartition) -> Vec<RelayTriadProfile> {
    let base = match partition {
        CorpusPartition::Development => 0x5100_0000_0000_0000,
        CorpusPartition::Calibration => 0x5200_0000_0000_0000,
        CorpusPartition::HeldOutEvaluation => 0x5300_0000_0000_0000,
        CorpusPartition::ExternalReplication => 0x5400_0000_0000_0000,
    };
    (0..relay_world_count(partition))
        .map(|index| RelayTriadProfile {
            schedule_revision: RELAY_TRIAD_SCHEDULE_REVISION,
            partition,
            index,
            seed: base + u64::from(index),
            mechanism_variant: (index % 2) as u8,
        })
        .collect()
}

pub(super) fn relay_scheduled_action(
    profile: RelayTriadProfile,
    legal_actions: &[PublicAction],
) -> Option<PublicAction> {
    (!legal_actions.is_empty())
        .then(|| legal_actions[usize::from(profile.index) % legal_actions.len()])
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RelayTriadState {
    bits: [bool; 3],
    public_to_hidden: [usize; 3],
    protected_hidden: usize,
    mechanism_variant: u8,
    step: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayTriadOracleSnapshot {
    world_digest: u64,
    state: RelayTriadState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayTriadCounterfactualOutcome {
    pub intervention: InterventionReceipt,
    pub public_outcome: PublicObservation,
    hidden_outcome_digest: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RelayTriadOracleError {
    SnapshotWorldMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayTriadActionReceipt {
    pub world_digest: u64,
    pub requested: PublicAction,
    pub status: ActionExecutionStatus,
    pub realized: Option<PublicAction>,
    pub pre_state: PublicObservation,
    pub post_state: Option<PublicObservation>,
    pub transition_digest: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayTriadEvaluator {
    world_digest: u64,
    state: RelayTriadState,
}

impl RelayTriadEvaluator {
    pub(super) fn build(profile: RelayTriadProfile) -> Self {
        let permutation = permutation3(profile.seed);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"eureka.relay-triad.identity.v1\0");
        encode_str(&mut bytes, RELAY_TRIAD_FAMILY_ID);
        bytes.extend_from_slice(&profile.seed.to_le_bytes());
        bytes.push(profile.mechanism_variant);
        bytes.push(partition_tag(profile.partition));
        bytes.extend(permutation.iter().map(|index| *index as u8));
        Self {
            world_digest: fnv1a64(&bytes),
            state: RelayTriadState {
                bits: [
                    profile.seed & 1 != 0,
                    profile.seed & 2 != 0,
                    profile.seed & 4 != 0,
                ],
                public_to_hidden: permutation,
                protected_hidden: (profile.seed as usize) % 3,
                mechanism_variant: profile.mechanism_variant,
                step: 0,
            },
        }
    }

    pub(super) fn world_digest(&self) -> u64 {
        self.world_digest
    }

    pub(super) fn hidden_state_digest(&self) -> u64 {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"eureka.relay-triad.state.v1\0");
        bytes.extend(self.state.bits.iter().map(|value| u8::from(*value)));
        bytes.extend_from_slice(&self.state.step.to_le_bytes());
        bytes.push(self.state.mechanism_variant);
        fnv1a64(&bytes)
    }

    pub(super) fn runtime(&mut self) -> RelayTriadRuntime<'_> {
        RelayTriadRuntime { inner: self }
    }

    pub(super) fn oracle_snapshot(&self) -> RelayTriadOracleSnapshot {
        RelayTriadOracleSnapshot {
            world_digest: self.world_digest,
            state: self.state.clone(),
        }
    }

    pub(super) fn counterfactual_from_snapshot(
        &self,
        snapshot: &RelayTriadOracleSnapshot,
        intervention: InterventionRequest,
        follow_up_action: PublicAction,
    ) -> Result<RelayTriadCounterfactualOutcome, RelayTriadOracleError> {
        if snapshot.world_digest != self.world_digest {
            return Err(RelayTriadOracleError::SnapshotWorldMismatch);
        }
        let mut branch = Self {
            world_digest: self.world_digest,
            state: snapshot.state.clone(),
        };
        let intervention = branch.apply_intervention(intervention);
        let public_outcome = branch.step(follow_up_action).observation;
        Ok(RelayTriadCounterfactualOutcome {
            intervention,
            public_outcome,
            hidden_outcome_digest: branch.hidden_state_digest(),
        })
    }

    pub(super) fn execute_qualified_action(
        &mut self,
        requested: PublicAction,
    ) -> RelayTriadActionReceipt {
        let pre_state = self.observe_public();
        if !self.legal_actions().contains(&requested) {
            return RelayTriadActionReceipt {
                world_digest: self.world_digest,
                requested,
                status: ActionExecutionStatus::RejectedIllegalAction,
                realized: None,
                pre_state,
                post_state: None,
                transition_digest: None,
            };
        }
        let post_state = self.step(requested).observation;
        RelayTriadActionReceipt {
            world_digest: self.world_digest,
            requested,
            status: ActionExecutionStatus::Applied,
            realized: Some(requested),
            transition_digest: Some(transition_digest(
                self.world_digest,
                &pre_state,
                requested,
                &post_state,
            )),
            pre_state,
            post_state: Some(post_state),
        }
    }

    fn observe_public(&self) -> PublicObservation {
        PublicObservation {
            step: self.state.step,
            fields: self
                .state
                .public_to_hidden
                .iter()
                .map(|hidden| PublicValue::Bit(self.state.bits[*hidden]))
                .collect(),
        }
    }

    fn legal_actions(&self) -> Vec<PublicAction> {
        vec![
            PublicAction::NoOp,
            PublicAction::Pulse { slot: 0 },
            PublicAction::Pulse { slot: 1 },
            PublicAction::Pulse { slot: 2 },
        ]
    }

    fn step(&mut self, action: PublicAction) -> StepReceipt {
        if let PublicAction::Pulse { slot } = action {
            if let Some(hidden) = self.state.public_to_hidden.get(slot as usize).copied() {
                self.state.bits[hidden] = !self.state.bits[hidden];
            }
        }
        let old = self.state.bits;
        let trigger = old[0];
        let relay = old[1];
        self.state.bits[1] = trigger;
        self.state.bits[2] = if self.state.mechanism_variant % 2 == 0 {
            relay
        } else {
            trigger ^ relay
        };
        self.state.step = self.state.step.saturating_add(1);
        StepReceipt {
            action,
            observation: self.observe_public(),
        }
    }

    fn apply_intervention(&mut self, request: InterventionRequest) -> InterventionReceipt {
        let Some(hidden) = self
            .state
            .public_to_hidden
            .get(request.slot as usize)
            .copied()
        else {
            return rejected(request, InterventionStatus::InvalidSlot);
        };
        let PublicValue::Bit(value) = request.value else {
            return rejected(request, InterventionStatus::UnsupportedValueType);
        };
        if hidden == self.state.protected_hidden {
            return rejected(request, InterventionStatus::RejectedByPolicy);
        }
        self.state.bits[hidden] = value;
        InterventionReceipt {
            requested: request,
            status: InterventionStatus::Applied,
            realized: Some(request),
        }
    }
}

pub(super) struct RelayTriadRuntime<'a> {
    inner: &'a mut RelayTriadEvaluator,
}

impl RelayTriadRuntime<'_> {
    pub(super) const fn public_schema_id(&self) -> &'static str {
        PUBLIC_SCHEMA_ID
    }

    pub(super) fn observe(&self) -> PublicObservation {
        self.inner.observe_public()
    }

    pub(super) fn legal_actions(&self) -> Vec<PublicAction> {
        self.inner.legal_actions()
    }

    pub(super) fn step(&mut self, action: PublicAction) -> StepReceipt {
        self.inner.step(action)
    }

    pub(super) fn intervene(&mut self, request: InterventionRequest) -> InterventionReceipt {
        self.inner.apply_intervention(request)
    }
}

fn rejected(request: InterventionRequest, status: InterventionStatus) -> InterventionReceipt {
    InterventionReceipt {
        requested: request,
        status,
        realized: None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RelayTriadTargetContractError {
    InsufficientObservationCapacity { required: usize, available: usize },
    InsufficientActionCapacity { required: usize, available: usize },
    PublicFieldCountMismatch { expected: usize, actual: usize },
    PublicFieldTypeMismatch { index: usize },
    UnsupportedPublicAction,
    TargetOutputDimensionMismatch { expected: usize, actual: usize },
    NonFiniteTargetOutput { index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RelayTriadFepTargetContract {
    scope: EurekaTargetScope,
    snapshot_replay_digest: u64,
    observation_dim: usize,
    action_count: usize,
    replay_digest: u64,
}

impl RelayTriadFepTargetContract {
    pub(super) fn new(
        snapshot: &FepEvaluationSnapshot,
    ) -> Result<Self, RelayTriadTargetContractError> {
        if snapshot.observation_dim() < 4 {
            return Err(RelayTriadTargetContractError::InsufficientObservationCapacity {
                required: 4,
                available: snapshot.observation_dim(),
            });
        }
        if snapshot.action_count() < 4 {
            return Err(RelayTriadTargetContractError::InsufficientActionCapacity {
                required: 4,
                available: snapshot.action_count(),
            });
        }
        let mut contract = Self {
            scope: EurekaTargetScope::ProductionFepComponentSnapshot,
            snapshot_replay_digest: snapshot.replay_digest(),
            observation_dim: snapshot.observation_dim(),
            action_count: snapshot.action_count(),
            replay_digest: 0,
        };
        contract.replay_digest = contract_digest(&contract);
        Ok(contract)
    }

    pub(super) fn replay_digest(&self) -> u64 {
        self.replay_digest
    }

    pub(super) fn snapshot_replay_digest(&self) -> u64 {
        self.snapshot_replay_digest
    }

    pub(super) fn encode_observation(
        &self,
        observation: &PublicObservation,
    ) -> Result<Vec<f64>, RelayTriadTargetContractError> {
        if observation.fields.len() != 3 {
            return Err(RelayTriadTargetContractError::PublicFieldCountMismatch {
                expected: 3,
                actual: observation.fields.len(),
            });
        }
        let mut encoded = vec![0.0; self.observation_dim];
        for (index, value) in observation.fields.iter().enumerate() {
            let PublicValue::Bit(bit) = value else {
                return Err(RelayTriadTargetContractError::PublicFieldTypeMismatch { index });
            };
            encoded[index] = if *bit { 1.0 } else { 0.0 };
        }
        encoded[3] = f64::from(observation.step);
        Ok(encoded)
    }

    pub(super) fn encode_action(
        &self,
        action: PublicAction,
    ) -> Result<usize, RelayTriadTargetContractError> {
        let target = match action {
            PublicAction::NoOp => 0,
            PublicAction::Pulse { slot: 0 } => 1,
            PublicAction::Pulse { slot: 1 } => 2,
            PublicAction::Pulse { slot: 2 } => 3,
            _ => return Err(RelayTriadTargetContractError::UnsupportedPublicAction),
        };
        Ok(target)
    }

    pub(super) fn decode_expected_fields(
        &self,
        expected_observation: &[f64],
    ) -> Result<Vec<PublicValue>, RelayTriadTargetContractError> {
        if expected_observation.len() != self.observation_dim {
            return Err(RelayTriadTargetContractError::TargetOutputDimensionMismatch {
                expected: self.observation_dim,
                actual: expected_observation.len(),
            });
        }
        if let Some(index) = expected_observation.iter().position(|value| !value.is_finite()) {
            return Err(RelayTriadTargetContractError::NonFiniteTargetOutput { index });
        }
        Ok(expected_observation[..3]
            .iter()
            .map(|value| PublicValue::Bit(*value >= 0.5))
            .collect())
    }
}

fn contract_digest(contract: &RelayTriadFepTargetContract) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, RELAY_TRIAD_TARGET_ADAPTER_REVISION);
    encode_str(&mut bytes, RELAY_TRIAD_FAMILY_ID);
    bytes.push(match contract.scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    bytes.extend_from_slice(&contract.snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&(contract.observation_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&(contract.action_count as u64).to_le_bytes());
    fnv1a64(&bytes)
}

fn transition_digest(
    world_digest: u64,
    pre: &PublicObservation,
    action: PublicAction,
    post: &PublicObservation,
) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.transition.v1\0");
    bytes.extend_from_slice(&world_digest.to_le_bytes());
    encode_observation(&mut bytes, pre);
    encode_action(&mut bytes, action);
    encode_observation(&mut bytes, post);
    fnv1a64(&bytes)
}

fn encode_observation(bytes: &mut Vec<u8>, observation: &PublicObservation) {
    bytes.extend_from_slice(&observation.step.to_le_bytes());
    bytes.extend_from_slice(&(observation.fields.len() as u64).to_le_bytes());
    for value in &observation.fields {
        match value {
            PublicValue::Bit(value) => {
                bytes.push(1);
                bytes.push(u8::from(*value));
            }
            PublicValue::Count(value) => {
                bytes.push(2);
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
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

fn permutation3(seed: u64) -> [usize; 3] {
    const P: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    P[(seed as usize) % P.len()]
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
    use crate::benchmarks::eureka::campaign_manifest::{
        scheduled_profiles, EUREKA_FIXTURE_FAMILIES,
    };
    use crate::benchmarks::eureka::hidden_world::{EvaluatorWorld, WorldBuildProfile};
    use std::collections::HashSet;
    use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, FepPredictionSession};

    fn profile(seed: u64, mechanism_variant: u8) -> RelayTriadProfile {
        RelayTriadProfile {
            schedule_revision: RELAY_TRIAD_SCHEDULE_REVISION,
            partition: CorpusPartition::HeldOutEvaluation,
            index: 0,
            seed,
            mechanism_variant,
        }
    }

    fn snapshot(obs_dim: usize, actions: usize) -> FepEvaluationSnapshot {
        let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim,
            num_actions: actions,
            enable_td_learning: true,
            ..Default::default()
        });
        FepPredictionSession::from_agent(&agent)
            .freeze_for_evaluation()
            .unwrap()
    }

    #[test]
    fn replay_and_qualified_transition_are_deterministic() {
        let mut a = RelayTriadEvaluator::build(profile(17, 0));
        let mut b = RelayTriadEvaluator::build(profile(17, 0));
        assert_eq!(a.world_digest(), b.world_digest());
        assert_eq!(a.runtime().observe(), b.runtime().observe());
        let action = PublicAction::Pulse { slot: 1 };
        assert_eq!(a.execute_qualified_action(action), b.execute_qualified_action(action));
    }

    #[test]
    fn legal_action_vocabulary_is_exact_and_invalid_pulse_does_not_advance() {
        let mut world = RelayTriadEvaluator::build(profile(5, 0));
        let expected = vec![
            PublicAction::NoOp,
            PublicAction::Pulse { slot: 0 },
            PublicAction::Pulse { slot: 1 },
            PublicAction::Pulse { slot: 2 },
        ];
        assert_eq!(world.runtime().legal_actions(), expected);
        let before = world.runtime().observe();
        let rejected = world.execute_qualified_action(PublicAction::Pulse { slot: 3 });
        assert_eq!(rejected.status, ActionExecutionStatus::RejectedIllegalAction);
        assert_eq!(world.runtime().observe(), before);
    }

    #[test]
    fn mechanism_variants_share_initial_surface_then_diverge() {
        // seed=3 gives hidden trigger=true, relay=true, so v0 output=relay
        // while v1 output=trigger XOR relay after the first NoOp.
        let mut a = RelayTriadEvaluator::build(profile(3, 0));
        let mut b = RelayTriadEvaluator::build(profile(3, 1));
        assert_eq!(a.runtime().observe(), b.runtime().observe());
        let _ = a.execute_qualified_action(PublicAction::NoOp);
        let _ = b.execute_qualified_action(PublicAction::NoOp);
        assert_ne!(a.runtime().observe(), b.runtime().observe());
    }

    #[test]
    fn counterfactual_branch_does_not_mutate_actual_world() {
        let world = RelayTriadEvaluator::build(profile(11, 0));
        let snapshot = world.oracle_snapshot();
        let before = world.observe_public();
        let hidden_before = world.hidden_state_digest();
        let outcome = world
            .counterfactual_from_snapshot(
                &snapshot,
                InterventionRequest {
                    slot: 1,
                    value: PublicValue::Bit(true),
                },
                PublicAction::NoOp,
            )
            .unwrap();
        assert_eq!(world.observe_public(), before);
        assert_eq!(world.hidden_state_digest(), hidden_before);
        assert_ne!(outcome.hidden_outcome_digest, 0);
    }

    #[test]
    fn schedule_is_balanced_and_seed_namespace_is_legacy_disjoint() {
        let mut relay_seeds = HashSet::new();
        for partition in [
            CorpusPartition::Development,
            CorpusPartition::Calibration,
            CorpusPartition::HeldOutEvaluation,
            CorpusPartition::ExternalReplication,
        ] {
            let profiles = relay_scheduled_profiles(partition);
            assert_eq!(profiles.len(), usize::from(relay_world_count(partition)));
            assert_eq!(
                profiles.iter().filter(|p| p.mechanism_variant == 0).count(),
                profiles.iter().filter(|p| p.mechanism_variant == 1).count()
            );
            relay_seeds.extend(profiles.iter().map(|p| p.seed));
        }
        for partition in [
            CorpusPartition::Development,
            CorpusPartition::Calibration,
            CorpusPartition::HeldOutEvaluation,
            CorpusPartition::ExternalReplication,
        ] {
            for family in EUREKA_FIXTURE_FAMILIES {
                for legacy in scheduled_profiles(partition, family) {
                    assert!(!relay_seeds.contains(&legacy.world.seed));
                }
            }
        }
    }

    #[test]
    fn sidecar_identity_is_distinct_from_legacy_identity() {
        let relay = RelayTriadEvaluator::build(profile(23, 0));
        let legacy = EvaluatorWorld::build(WorldBuildProfile {
            family: EUREKA_FIXTURE_FAMILIES[1],
            seed: 23,
            mechanism_variant: 0,
            partition: CorpusPartition::HeldOutEvaluation,
        });
        assert_ne!(relay.world_digest(), legacy.world_digest());
    }

    #[test]
    fn four_by_four_target_accepts_relay_and_mapping_is_lossless() {
        let frozen = snapshot(4, 4);
        let contract = RelayTriadFepTargetContract::new(&frozen).unwrap();
        let observation = PublicObservation {
            step: 7,
            fields: vec![
                PublicValue::Bit(true),
                PublicValue::Bit(false),
                PublicValue::Bit(true),
            ],
        };
        assert_eq!(
            contract.encode_observation(&observation).unwrap(),
            vec![1.0, 0.0, 1.0, 7.0]
        );
        assert_eq!(contract.encode_action(PublicAction::NoOp), Ok(0));
        assert_eq!(contract.encode_action(PublicAction::Pulse { slot: 0 }), Ok(1));
        assert_eq!(contract.encode_action(PublicAction::Pulse { slot: 1 }), Ok(2));
        assert_eq!(contract.encode_action(PublicAction::Pulse { slot: 2 }), Ok(3));
        assert_eq!(
            contract.encode_action(PublicAction::Pulse { slot: 3 }),
            Err(RelayTriadTargetContractError::UnsupportedPublicAction)
        );
        assert_eq!(
            contract.decode_expected_fields(&[0.49, 0.5, 0.9, 99.0]).unwrap(),
            vec![
                PublicValue::Bit(false),
                PublicValue::Bit(true),
                PublicValue::Bit(true),
            ]
        );
    }

    #[test]
    fn insufficient_target_dimensions_fail_closed() {
        assert!(matches!(
            RelayTriadFepTargetContract::new(&snapshot(3, 4)),
            Err(RelayTriadTargetContractError::InsufficientObservationCapacity {
                required: 4,
                available: 3,
            })
        ));
        assert!(matches!(
            RelayTriadFepTargetContract::new(&snapshot(4, 3)),
            Err(RelayTriadTargetContractError::InsufficientActionCapacity {
                required: 4,
                available: 3,
            })
        ));
    }

    #[test]
    fn target_contract_identity_binds_learned_snapshot() {
        let mut agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 4,
            enable_td_learning: true,
            ..Default::default()
        });
        let a = FepPredictionSession::from_agent(&agent)
            .freeze_for_evaluation()
            .unwrap();
        agent.model.transition_matrices[0][0][0] += 0.01;
        let b = FepPredictionSession::from_agent(&agent)
            .freeze_for_evaluation()
            .unwrap();
        let ca = RelayTriadFepTargetContract::new(&a).unwrap();
        let cb = RelayTriadFepTargetContract::new(&b).unwrap();
        assert_ne!(ca.snapshot_replay_digest(), cb.snapshot_replay_digest());
        assert_ne!(ca.replay_digest(), cb.replay_digest());
    }
}
