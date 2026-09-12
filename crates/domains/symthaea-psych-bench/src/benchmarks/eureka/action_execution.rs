// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evaluator-owned action realization boundary for EUREKA hidden worlds.
//!
//! A target request is not proof that an action was accepted or realized.
//! Qualification code should consume [`QualifiedActionReceipt`] rather than
//! infer realization from a request echo.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2090>

use super::hidden_world::{EvaluatorWorld, PublicAction, PublicObservation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ActionExecutionStatus {
    Applied,
    RejectedIllegalAction,
}

/// Evaluator-owned realization evidence for one synthetic-world action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QualifiedActionReceipt {
    pub world_digest: u64,
    pub requested: PublicAction,
    pub status: ActionExecutionStatus,
    pub realized: Option<PublicAction>,
    pub pre_state: PublicObservation,
    pub post_state: Option<PublicObservation>,
    pub transition_digest: Option<u64>,
}

impl EvaluatorWorld {
    /// Canonical action route for EUREKA qualification.
    ///
    /// Illegal/wrong-family actions are rejected before `step()` is called,
    /// therefore they do not advance the canonical evaluator state.
    pub(super) fn execute_qualified_action(
        &mut self,
        requested: PublicAction,
    ) -> QualifiedActionReceipt {
        let world_digest = self.world_digest();
        let pre_state = self.runtime().observe();
        let legal_actions = self.runtime().legal_actions();

        if !legal_actions.contains(&requested) {
            return QualifiedActionReceipt {
                world_digest,
                requested,
                status: ActionExecutionStatus::RejectedIllegalAction,
                realized: None,
                pre_state,
                post_state: None,
                transition_digest: None,
            };
        }

        let step = self.runtime().step(requested);
        let transition_digest = digest_transition(world_digest, &pre_state, &step.observation, requested);
        QualifiedActionReceipt {
            world_digest,
            requested,
            status: ActionExecutionStatus::Applied,
            realized: Some(requested),
            pre_state,
            post_state: Some(step.observation),
            transition_digest: Some(transition_digest),
        }
    }
}

fn digest_transition(
    world_digest: u64,
    pre: &PublicObservation,
    post: &PublicObservation,
    action: PublicAction,
) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.realized-action-transition.v1\0");
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
            super::hidden_world::PublicValue::Bit(value) => {
                bytes.push(1);
                bytes.push(u8::from(*value));
            }
            super::hidden_world::PublicValue::Count(value) => {
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
    use crate::benchmarks::eureka::hidden_world::{
        CorpusPartition, FixtureFamily, WorldBuildProfile,
    };

    fn world(family: FixtureFamily, seed: u64) -> EvaluatorWorld {
        EvaluatorWorld::build(WorldBuildProfile {
            family,
            seed,
            mechanism_variant: 0,
            partition: CorpusPartition::HeldOutEvaluation,
        })
    }

    #[test]
    fn legal_action_is_applied_and_bound_to_fresh_state() {
        let mut evaluator = world(FixtureFamily::CausalBits, 3);
        let before = evaluator.runtime().observe();
        let receipt = evaluator.execute_qualified_action(PublicAction::NoOp);
        assert_eq!(receipt.status, ActionExecutionStatus::Applied);
        assert_eq!(receipt.realized, Some(PublicAction::NoOp));
        assert_eq!(receipt.pre_state, before);
        let post = receipt.post_state.expect("applied action has post-state");
        assert!(post.step > before.step);
        assert_ne!(receipt.transition_digest, None);
    }

    #[test]
    fn wrong_family_action_is_rejected_without_state_advance() {
        let mut evaluator = world(FixtureFamily::CausalBits, 3);
        let before = evaluator.runtime().observe();
        let requested = PublicAction::Transfer {
            from: 0,
            to: 1,
            amount: 1,
        };
        let receipt = evaluator.execute_qualified_action(requested);
        assert_eq!(receipt.status, ActionExecutionStatus::RejectedIllegalAction);
        assert_eq!(receipt.requested, requested);
        assert_eq!(receipt.realized, None);
        assert_eq!(receipt.post_state, None);
        assert_eq!(receipt.transition_digest, None);
        assert_eq!(evaluator.runtime().observe(), before);
    }

    #[test]
    fn out_of_range_pulse_is_rejected_without_state_advance() {
        let mut evaluator = world(FixtureFamily::CausalBits, 3);
        let before = evaluator.runtime().observe();
        let receipt = evaluator.execute_qualified_action(PublicAction::Pulse { slot: 99 });
        assert_eq!(receipt.status, ActionExecutionStatus::RejectedIllegalAction);
        assert_eq!(receipt.realized, None);
        assert_eq!(evaluator.runtime().observe(), before);
    }

    #[test]
    fn resource_world_rejects_pulse_instead_of_silent_noop() {
        let mut evaluator = world(FixtureFamily::ResourceFlow, 8);
        let before = evaluator.runtime().observe();
        let receipt = evaluator.execute_qualified_action(PublicAction::Pulse { slot: 0 });
        assert_eq!(receipt.status, ActionExecutionStatus::RejectedIllegalAction);
        assert_eq!(receipt.realized, None);
        assert_eq!(evaluator.runtime().observe(), before);
    }

    #[test]
    fn changing_action_changes_transition_identity() {
        let mut a = world(FixtureFamily::CausalBits, 3);
        let mut b = world(FixtureFamily::CausalBits, 3);
        let no_op = a.execute_qualified_action(PublicAction::NoOp);
        let pulse = b.execute_qualified_action(PublicAction::Pulse { slot: 0 });
        assert_eq!(no_op.world_digest, pulse.world_digest);
        assert_ne!(no_op.transition_digest, pulse.transition_digest);
    }

    #[test]
    fn receipt_world_lineage_differs_across_worlds() {
        let mut a = world(FixtureFamily::CausalBits, 3);
        let mut b = world(FixtureFamily::CausalBits, 4);
        let receipt_a = a.execute_qualified_action(PublicAction::NoOp);
        let receipt_b = b.execute_qualified_action(PublicAction::NoOp);
        assert_ne!(receipt_a.world_digest, receipt_b.world_digest);
    }

    #[test]
    fn no_op_is_explicitly_realized_when_accepted() {
        let mut evaluator = world(FixtureFamily::ResourceFlow, 8);
        let receipt = evaluator.execute_qualified_action(PublicAction::NoOp);
        assert_eq!(receipt.status, ActionExecutionStatus::Applied);
        assert_eq!(receipt.requested, PublicAction::NoOp);
        assert_eq!(receipt.realized, Some(PublicAction::NoOp));
    }
}
