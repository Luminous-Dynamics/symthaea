// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evaluator-owned hidden worlds for EUREKA qualification.
//!
//! The target receives only [`RuntimeWorld`]. Exact world identity, hidden
//! state, structural roles, evaluator snapshots, and counterfactual outcomes
//! remain evaluator-side. This module is benchmark infrastructure, not a new
//! production world model.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2048>

/// Shared public schema for the first synthetic fixtures. The schema does not
/// encode which hidden mechanism generated an episode.
pub const PUBLIC_SCHEMA_ID: &str = "eureka.hidden-world.generic-fields.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum CorpusPartition {
    Development,
    Calibration,
    HeldOutEvaluation,
    ExternalReplication,
}

/// Values allowed across the target-facing observation/intervention boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PublicValue {
    Bit(bool),
    Count(i32),
}

/// Target-visible state. Public field order is intentionally opaque.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicObservation {
    pub step: u32,
    pub fields: Vec<PublicValue>,
}

/// Minimal target-visible action vocabulary for the proving fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PublicAction {
    NoOp,
    Pulse { slot: u8 },
    Transfer { from: u8, to: u8, amount: i16 },
}

/// Requested treatment. A request is not evidence that treatment occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterventionRequest {
    pub slot: u8,
    pub value: PublicValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterventionStatus {
    Applied,
    RejectedByPolicy,
    UnsupportedValueType,
    InvalidSlot,
}

/// Separates requested assignment from realized intervention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterventionReceipt {
    pub requested: InterventionRequest,
    pub status: InterventionStatus,
    pub realized: Option<InterventionRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepReceipt {
    pub action: PublicAction,
    pub observation: PublicObservation,
}

/// Canonical target interface. There is deliberately no catch-all extension
/// query through which hidden evaluator truth can be requested.
pub struct RuntimeWorld<'a> {
    inner: &'a mut EvaluatorWorld,
}

impl RuntimeWorld<'_> {
    pub const fn public_schema_id(&self) -> &'static str {
        PUBLIC_SCHEMA_ID
    }

    pub fn observe(&self) -> PublicObservation {
        self.inner.observe_public()
    }

    pub fn legal_actions(&self) -> Vec<PublicAction> {
        self.inner.legal_actions()
    }

    pub fn step(&mut self, action: PublicAction) -> StepReceipt {
        self.inner.step(action)
    }

    pub fn intervene(&mut self, request: InterventionRequest) -> InterventionReceipt {
        self.inner.apply_intervention(request)
    }

    pub fn snapshot_public_state(&self) -> PublicObservation {
        self.observe()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum FixtureFamily {
    CausalBits,
    ResourceFlow,
}

/// Evaluator-only profile. `mechanism_variant` changes hidden dynamics without
/// changing the public schema or initial state for the same seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct WorldBuildProfile {
    pub family: FixtureFamily,
    pub seed: u64,
    pub mechanism_variant: u8,
    pub partition: CorpusPartition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorldIdentity {
    family: FixtureFamily,
    seed: u64,
    mechanism_variant: u8,
    partition: CorpusPartition,
    permutation: Vec<usize>,
    digest: u64,
}

impl WorldIdentity {
    fn new(profile: WorldBuildProfile, permutation: Vec<usize>) -> Self {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(b"eureka.hidden-world.identity.v1\0");
        bytes.push(match profile.family {
            FixtureFamily::CausalBits => 1,
            FixtureFamily::ResourceFlow => 2,
        });
        bytes.extend_from_slice(&profile.seed.to_le_bytes());
        bytes.push(profile.mechanism_variant);
        bytes.push(match profile.partition {
            CorpusPartition::Development => 1,
            CorpusPartition::Calibration => 2,
            CorpusPartition::HeldOutEvaluation => 3,
            CorpusPartition::ExternalReplication => 4,
        });
        bytes.extend(permutation.iter().map(|index| *index as u8));
        let digest = fnv1a64(&bytes);
        Self {
            family: profile.family,
            seed: profile.seed,
            mechanism_variant: profile.mechanism_variant,
            partition: profile.partition,
            permutation,
            digest,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WorldState {
    CausalBits(CausalBitsState),
    ResourceFlow(ResourceFlowState),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CausalBitsState {
    bits: [bool; 4],
    public_to_hidden: [usize; 4],
    protected_hidden: usize,
    mechanism_variant: u8,
    step: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResourceFlowState {
    stocks: [i32; 3],
    public_to_hidden: [usize; 3],
    protected_hidden: usize,
    mechanism_variant: u8,
    step: u32,
}

/// Evaluator-only exact actual-world snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OracleSnapshot {
    world_digest: u64,
    state: WorldState,
}

/// Evaluator-only hypothetical result. This is never a runtime observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OracleCounterfactualOutcome {
    pub intervention: InterventionReceipt,
    pub public_outcome: PublicObservation,
    hidden_outcome_digest: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OracleError {
    SnapshotWorldMismatch,
}

/// Owns evaluator truth and lends the candidate only a [`RuntimeWorld`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EvaluatorWorld {
    identity: WorldIdentity,
    state: WorldState,
}

impl EvaluatorWorld {
    pub(super) fn build(profile: WorldBuildProfile) -> Self {
        let state = match profile.family {
            FixtureFamily::CausalBits => {
                let public_to_hidden = permutation4(profile.seed);
                WorldState::CausalBits(CausalBitsState {
                    bits: [
                        profile.seed & 1 != 0,
                        profile.seed & 2 != 0,
                        profile.seed & 4 != 0,
                        profile.seed & 8 != 0,
                    ],
                    public_to_hidden,
                    protected_hidden: (profile.seed as usize) % 4,
                    mechanism_variant: profile.mechanism_variant,
                    step: 0,
                })
            }
            FixtureFamily::ResourceFlow => {
                let public_to_hidden = permutation3(profile.seed);
                WorldState::ResourceFlow(ResourceFlowState {
                    stocks: [
                        2 + (profile.seed % 3) as i32,
                        1 + ((profile.seed >> 2) % 3) as i32,
                        ((profile.seed >> 4) % 2) as i32,
                    ],
                    public_to_hidden,
                    protected_hidden: (profile.seed as usize) % 3,
                    mechanism_variant: profile.mechanism_variant,
                    step: 0,
                })
            }
        };
        let permutation = match &state {
            WorldState::CausalBits(s) => s.public_to_hidden.to_vec(),
            WorldState::ResourceFlow(s) => s.public_to_hidden.to_vec(),
        };
        Self {
            identity: WorldIdentity::new(profile, permutation),
            state,
        }
    }

    pub(super) fn runtime(&mut self) -> RuntimeWorld<'_> {
        RuntimeWorld { inner: self }
    }

    pub(super) fn world_digest(&self) -> u64 {
        self.identity.digest
    }

    pub(super) fn hidden_state_digest(&self) -> u64 {
        digest_state(&self.state)
    }

    pub(super) fn hidden_structural_roles(&self) -> Vec<&'static str> {
        match &self.state {
            WorldState::CausalBits(_) => vec!["source", "mediator", "downstream", "inhibitor"],
            WorldState::ResourceFlow(_) => vec!["source_stock", "buffer_stock", "sink_stock"],
        }
    }

    pub(super) fn oracle_snapshot(&self) -> OracleSnapshot {
        OracleSnapshot {
            world_digest: self.identity.digest,
            state: self.state.clone(),
        }
    }

    pub(super) fn counterfactual_from_snapshot(
        &self,
        snapshot: &OracleSnapshot,
        intervention: InterventionRequest,
        follow_up_action: PublicAction,
    ) -> Result<OracleCounterfactualOutcome, OracleError> {
        if snapshot.world_digest != self.identity.digest {
            return Err(OracleError::SnapshotWorldMismatch);
        }
        let mut branch = Self {
            identity: self.identity.clone(),
            state: snapshot.state.clone(),
        };
        let intervention = branch.apply_intervention(intervention);
        let public_outcome = branch.step(follow_up_action).observation;
        Ok(OracleCounterfactualOutcome {
            intervention,
            public_outcome,
            hidden_outcome_digest: branch.hidden_state_digest(),
        })
    }

    fn observe_public(&self) -> PublicObservation {
        match &self.state {
            WorldState::CausalBits(s) => PublicObservation {
                step: s.step,
                fields: s
                    .public_to_hidden
                    .iter()
                    .map(|hidden| PublicValue::Bit(s.bits[*hidden]))
                    .collect(),
            },
            WorldState::ResourceFlow(s) => PublicObservation {
                step: s.step,
                fields: s
                    .public_to_hidden
                    .iter()
                    .map(|hidden| PublicValue::Count(s.stocks[*hidden]))
                    .collect(),
            },
        }
    }

    fn legal_actions(&self) -> Vec<PublicAction> {
        match &self.state {
            WorldState::CausalBits(_) => {
                let mut actions = vec![PublicAction::NoOp];
                actions.extend((0..4).map(|slot| PublicAction::Pulse { slot }));
                actions
            }
            WorldState::ResourceFlow(_) => vec![
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
            ],
        }
    }

    fn step(&mut self, action: PublicAction) -> StepReceipt {
        match &mut self.state {
            WorldState::CausalBits(s) => step_causal_bits(s, action),
            WorldState::ResourceFlow(s) => step_resource_flow(s, action),
        }
        StepReceipt {
            action,
            observation: self.observe_public(),
        }
    }

    fn apply_intervention(&mut self, request: InterventionRequest) -> InterventionReceipt {
        let status = match &mut self.state {
            WorldState::CausalBits(s) => {
                let Some(hidden) = s.public_to_hidden.get(request.slot as usize).copied() else {
                    return rejected(request, InterventionStatus::InvalidSlot);
                };
                let PublicValue::Bit(value) = request.value else {
                    return rejected(request, InterventionStatus::UnsupportedValueType);
                };
                if hidden == s.protected_hidden {
                    InterventionStatus::RejectedByPolicy
                } else {
                    s.bits[hidden] = value;
                    InterventionStatus::Applied
                }
            }
            WorldState::ResourceFlow(s) => {
                let Some(hidden) = s.public_to_hidden.get(request.slot as usize).copied() else {
                    return rejected(request, InterventionStatus::InvalidSlot);
                };
                let PublicValue::Count(value) = request.value else {
                    return rejected(request, InterventionStatus::UnsupportedValueType);
                };
                if hidden == s.protected_hidden {
                    InterventionStatus::RejectedByPolicy
                } else {
                    s.stocks[hidden] = value.max(0);
                    InterventionStatus::Applied
                }
            }
        };
        InterventionReceipt {
            requested: request,
            status,
            realized: (status == InterventionStatus::Applied).then_some(request),
        }
    }
}

fn rejected(request: InterventionRequest, status: InterventionStatus) -> InterventionReceipt {
    InterventionReceipt {
        requested: request,
        status,
        realized: None,
    }
}

fn step_causal_bits(state: &mut CausalBitsState, action: PublicAction) {
    if let PublicAction::Pulse { slot } = action {
        if let Some(hidden) = state.public_to_hidden.get(slot as usize).copied() {
            state.bits[hidden] = !state.bits[hidden];
        }
    }
    let old = state.bits;
    let source = old[0];
    let mediator = old[1];
    let inhibitor = old[3];
    state.bits[1] = source;
    state.bits[2] = if state.mechanism_variant % 2 == 0 {
        mediator && !inhibitor
    } else {
        (source ^ mediator) && !inhibitor
    };
    state.step = state.step.saturating_add(1);
}

fn step_resource_flow(state: &mut ResourceFlowState, action: PublicAction) {
    if let PublicAction::Transfer { from, to, amount } = action {
        let from_hidden = state.public_to_hidden.get(from as usize).copied();
        let to_hidden = state.public_to_hidden.get(to as usize).copied();
        if let (Some(from_hidden), Some(to_hidden)) = (from_hidden, to_hidden) {
            let requested = i32::from(amount.max(0));
            let moved = requested.min(state.stocks[from_hidden].max(0));
            state.stocks[from_hidden] -= moved;
            state.stocks[to_hidden] += moved;
        }
    }
    if state.mechanism_variant % 2 == 0 {
        state.stocks[0] = state.stocks[0].saturating_add(1);
    } else if state.stocks[1] > 0 {
        state.stocks[1] -= 1;
        state.stocks[2] = state.stocks[2].saturating_add(1);
    }
    state.step = state.step.saturating_add(1);
}

fn permutation4(seed: u64) -> [usize; 4] {
    const P: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
        [1, 3, 0, 2],
        [2, 0, 3, 1],
    ];
    P[(seed as usize) % P.len()]
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

fn digest_state(state: &WorldState) -> u64 {
    let mut bytes = Vec::with_capacity(32);
    match state {
        WorldState::CausalBits(s) => {
            bytes.push(1);
            bytes.extend(s.bits.iter().map(|value| u8::from(*value)));
            bytes.extend_from_slice(&s.step.to_le_bytes());
            bytes.push(s.mechanism_variant);
        }
        WorldState::ResourceFlow(s) => {
            bytes.push(2);
            for stock in s.stocks {
                bytes.extend_from_slice(&stock.to_le_bytes());
            }
            bytes.extend_from_slice(&s.step.to_le_bytes());
            bytes.push(s.mechanism_variant);
        }
    }
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

    fn profile(family: FixtureFamily, seed: u64, mechanism_variant: u8) -> WorldBuildProfile {
        WorldBuildProfile {
            family,
            seed,
            mechanism_variant,
            partition: CorpusPartition::HeldOutEvaluation,
        }
    }

    #[test]
    fn two_world_families_replay_deterministically() {
        for family in [FixtureFamily::CausalBits, FixtureFamily::ResourceFlow] {
            let mut a = EvaluatorWorld::build(profile(family, 17, 0));
            let mut b = EvaluatorWorld::build(profile(family, 17, 0));
            assert_eq!(a.world_digest(), b.world_digest());
            assert_eq!(a.observe_public(), b.observe_public());
            assert_eq!(a.legal_actions(), b.legal_actions());
            let action = a.legal_actions()[0];
            assert_eq!(a.step(action), b.step(action));
            assert_eq!(a.hidden_state_digest(), b.hidden_state_digest());
        }
    }

    #[test]
    fn runtime_observation_exposes_neither_world_digest_nor_structural_roles() {
        let mut evaluator = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 9, 0));
        let private_digest = evaluator.world_digest();
        let private_roles = evaluator.hidden_structural_roles();
        let runtime = evaluator.runtime();
        assert_eq!(runtime.public_schema_id(), PUBLIC_SCHEMA_ID);
        let public = format!("{:?}", runtime.observe());
        assert!(!public.contains(&private_digest.to_string()));
        for role in private_roles {
            assert!(!public.contains(role));
        }
    }

    #[test]
    fn hidden_mechanism_variants_share_initial_public_surface_then_diverge() {
        let mut a = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 3, 0));
        let mut b = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 3, 1));
        assert_ne!(a.world_digest(), b.world_digest());
        assert_eq!(a.observe_public(), b.observe_public());
        assert_eq!(a.runtime().public_schema_id(), b.runtime().public_schema_id());
        let _ = a.runtime().step(PublicAction::NoOp);
        let _ = b.runtime().step(PublicAction::NoOp);
        assert_ne!(a.observe_public(), b.observe_public());
    }

    #[test]
    fn requested_and_realized_intervention_are_distinct() {
        let mut evaluator = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 0, 0));
        let request = InterventionRequest {
            slot: 0,
            value: PublicValue::Bit(true),
        };
        let receipt = evaluator.runtime().intervene(request);
        assert_eq!(receipt.requested, request);
        assert_eq!(receipt.status, InterventionStatus::RejectedByPolicy);
        assert_eq!(receipt.realized, None);
    }

    #[test]
    fn invalid_and_wrong_typed_interventions_fail_explicitly() {
        let mut evaluator = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 5, 0));
        let invalid = evaluator.runtime().intervene(InterventionRequest {
            slot: 99,
            value: PublicValue::Bit(true),
        });
        assert_eq!(invalid.status, InterventionStatus::InvalidSlot);
        let wrong_type = evaluator.runtime().intervene(InterventionRequest {
            slot: 0,
            value: PublicValue::Count(1),
        });
        assert_eq!(wrong_type.status, InterventionStatus::UnsupportedValueType);
    }

    #[test]
    fn counterfactual_branch_cannot_mutate_actual_episode() {
        let evaluator = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 11, 0));
        let snapshot = evaluator.oracle_snapshot();
        let before_public = evaluator.observe_public();
        let before_hidden = evaluator.hidden_state_digest();
        let outcome = evaluator
            .counterfactual_from_snapshot(
                &snapshot,
                InterventionRequest {
                    slot: 0,
                    value: PublicValue::Bit(true),
                },
                PublicAction::NoOp,
            )
            .expect("snapshot belongs to evaluator");
        assert_eq!(evaluator.observe_public(), before_public);
        assert_eq!(evaluator.hidden_state_digest(), before_hidden);
        assert_ne!(outcome.hidden_outcome_digest, 0);
    }

    #[test]
    fn cross_world_counterfactual_snapshot_is_rejected() {
        let a = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 1, 0));
        let b = EvaluatorWorld::build(profile(FixtureFamily::CausalBits, 2, 0));
        let snapshot = a.oracle_snapshot();
        assert_eq!(
            b.counterfactual_from_snapshot(
                &snapshot,
                InterventionRequest {
                    slot: 0,
                    value: PublicValue::Bit(false),
                },
                PublicAction::NoOp,
            ),
            Err(OracleError::SnapshotWorldMismatch)
        );
    }

    #[test]
    fn corpus_partition_changes_evaluator_identity_not_public_state() {
        let mut development = EvaluatorWorld::build(WorldBuildProfile {
            family: FixtureFamily::ResourceFlow,
            seed: 21,
            mechanism_variant: 0,
            partition: CorpusPartition::Development,
        });
        let mut held_out = EvaluatorWorld::build(WorldBuildProfile {
            family: FixtureFamily::ResourceFlow,
            seed: 21,
            mechanism_variant: 0,
            partition: CorpusPartition::HeldOutEvaluation,
        });
        assert_ne!(development.world_digest(), held_out.world_digest());
        assert_eq!(development.observe_public(), held_out.observe_public());
        assert_eq!(
            development.runtime().public_schema_id(),
            held_out.runtime().public_schema_id()
        );
    }

    #[test]
    fn successful_intervention_reports_realized_treatment() {
        let mut evaluator = EvaluatorWorld::build(profile(FixtureFamily::ResourceFlow, 0, 0));
        let request = InterventionRequest {
            slot: 1,
            value: PublicValue::Count(9),
        };
        let receipt = evaluator.runtime().intervene(request);
        assert_eq!(receipt.status, InterventionStatus::Applied);
        assert_eq!(receipt.realized, Some(request));
        assert_eq!(evaluator.observe_public().fields[1], PublicValue::Count(9));
    }

    #[test]
    fn all_partition_variants_are_constructible() {
        for partition in [
            CorpusPartition::Development,
            CorpusPartition::Calibration,
            CorpusPartition::HeldOutEvaluation,
            CorpusPartition::ExternalReplication,
        ] {
            let world = EvaluatorWorld::build(WorldBuildProfile {
                family: FixtureFamily::CausalBits,
                seed: 7,
                mechanism_variant: 0,
                partition,
            });
            assert_ne!(world.world_digest(), 0);
        }
    }
}
