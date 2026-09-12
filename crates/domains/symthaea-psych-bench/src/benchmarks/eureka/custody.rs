// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prospective custody for EUREKA-002 consequence trials.
//!
//! A prediction must be frozen before the fresh transition outcome is revealed.
//! The frozen object is intentionally move-only (no `Clone`) and is consumed
//! exactly once when scoring. Evaluator logical sequence proves local ordering;
//! it is not claimed to be trusted wall-clock time.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2087>

use super::consequence::{
    ConsequencePrediction, ConsequenceScore, ConsequenceScoringError, copy_current_state_baseline,
    score_consequence,
};
use super::hidden_world::{PublicAction, PublicObservation, PublicValue, StepReceipt};

pub(super) const COPY_CURRENT_STATE_BASELINE_ID: &str = "EUREKA.BASELINE.COPY_CURRENT_STATE.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CustodyError {
    IllegalActionAtFreeze,
    WrongEvaluatorWorld,
    ActionMismatchAtReveal,
    NonAdvancingOutcome,
    SequenceExhausted,
    Scoring(ConsequenceScoringError),
}

/// Exact frozen prediction custody object. Deliberately not `Clone`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct FrozenPrediction {
    world_digest: u64,
    commit_sequence: u64,
    pre_state: PublicObservation,
    prediction: ConsequencePrediction,
    commitment_digest: u64,
}

impl FrozenPrediction {
    pub(super) fn commitment_digest(&self) -> u64 {
        self.commitment_digest
    }

    pub(super) fn commit_sequence(&self) -> u64 {
        self.commit_sequence
    }
}

/// One candidate-vs-copy comparison on the exact same transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MatchedConsequenceTrial {
    pub world_digest: u64,
    pub commitment_digest: u64,
    pub transition_digest: u64,
    pub commit_sequence: u64,
    pub reveal_sequence: u64,
    pub action: PublicAction,
    pub candidate_score: ConsequenceScore,
    pub copy_baseline_score: ConsequenceScore,
    pub baseline_id: &'static str,
}

/// Evaluator-owned custody state. The target never owns this object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PredictionCustodian {
    world_digest: u64,
    next_sequence: u64,
}

impl PredictionCustodian {
    pub(super) fn new(world_digest: u64) -> Self {
        Self {
            world_digest,
            next_sequence: 1,
        }
    }

    /// Freeze an exact candidate prediction while only pre-outcome information
    /// is available. The legal-action set is captured from the same public
    /// pre-state and prevents arbitrary unsupported action requests from
    /// entering a positive EUREKA-002 trial.
    pub(super) fn freeze(
        &mut self,
        pre_state: &PublicObservation,
        legal_actions: &[PublicAction],
        prediction: &ConsequencePrediction,
    ) -> Result<FrozenPrediction, CustodyError> {
        if !legal_actions.contains(&prediction.action) {
            return Err(CustodyError::IllegalActionAtFreeze);
        }
        let commit_sequence = self.take_sequence()?;
        let pre_state = pre_state.clone();
        let prediction = prediction.clone();
        let commitment_digest = commitment_digest(
            self.world_digest,
            commit_sequence,
            &pre_state,
            &prediction,
        );
        Ok(FrozenPrediction {
            world_digest: self.world_digest,
            commit_sequence,
            pre_state,
            prediction,
            commitment_digest,
        })
    }

    /// Consume one frozen prediction after the fresh evaluator transition is
    /// available and score both candidate and copy baseline on that exact same
    /// pre-state/action/outcome tuple.
    pub(super) fn score_after_reveal(
        &mut self,
        frozen: FrozenPrediction,
        fresh_transition: &StepReceipt,
    ) -> Result<MatchedConsequenceTrial, CustodyError> {
        if frozen.world_digest != self.world_digest {
            return Err(CustodyError::WrongEvaluatorWorld);
        }
        if frozen.prediction.action != fresh_transition.action {
            return Err(CustodyError::ActionMismatchAtReveal);
        }
        if fresh_transition.observation.step <= frozen.pre_state.step {
            return Err(CustodyError::NonAdvancingOutcome);
        }

        let reveal_sequence = self.take_sequence()?;
        let candidate_score = score_consequence(
            &frozen.pre_state,
            &frozen.prediction,
            &fresh_transition.observation,
        )
        .map_err(CustodyError::Scoring)?;

        let copy = copy_current_state_baseline(&frozen.pre_state, fresh_transition.action);
        let copy_baseline_score = score_consequence(
            &frozen.pre_state,
            &copy,
            &fresh_transition.observation,
        )
        .map_err(CustodyError::Scoring)?;

        let transition_digest = transition_digest(
            self.world_digest,
            &frozen.pre_state,
            fresh_transition,
        );

        Ok(MatchedConsequenceTrial {
            world_digest: self.world_digest,
            commitment_digest: frozen.commitment_digest,
            transition_digest,
            commit_sequence: frozen.commit_sequence,
            reveal_sequence,
            action: fresh_transition.action,
            candidate_score,
            copy_baseline_score,
            baseline_id: COPY_CURRENT_STATE_BASELINE_ID,
        })
    }

    fn take_sequence(&mut self) -> Result<u64, CustodyError> {
        let current = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(CustodyError::SequenceExhausted)?;
        Ok(current)
    }
}

fn commitment_digest(
    world_digest: u64,
    sequence: u64,
    pre: &PublicObservation,
    prediction: &ConsequencePrediction,
) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.prediction-commitment.v1\0");
    bytes.extend_from_slice(&world_digest.to_le_bytes());
    bytes.extend_from_slice(&sequence.to_le_bytes());
    encode_observation(&mut bytes, pre);
    encode_prediction(&mut bytes, prediction);
    fnv1a64(&bytes)
}

fn transition_digest(
    world_digest: u64,
    pre: &PublicObservation,
    fresh: &StepReceipt,
) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.transition.v1\0");
    bytes.extend_from_slice(&world_digest.to_le_bytes());
    encode_observation(&mut bytes, pre);
    encode_action(&mut bytes, fresh.action);
    encode_observation(&mut bytes, &fresh.observation);
    fnv1a64(&bytes)
}

fn encode_prediction(bytes: &mut Vec<u8>, prediction: &ConsequencePrediction) {
    encode_action(bytes, prediction.action);
    match &prediction.outcome {
        super::consequence::PredictionOutcome::Predicted { fields } => {
            bytes.push(1);
            bytes.extend_from_slice(&(fields.len() as u64).to_le_bytes());
            for value in fields {
                encode_value(bytes, *value);
            }
        }
        super::consequence::PredictionOutcome::AbstainInsufficientEvidence => bytes.push(2),
        super::consequence::PredictionOutcome::OutOfQualifiedDomain => bytes.push(3),
    }
}

fn encode_observation(bytes: &mut Vec<u8>, observation: &PublicObservation) {
    bytes.extend_from_slice(&observation.step.to_le_bytes());
    bytes.extend_from_slice(&(observation.fields.len() as u64).to_le_bytes());
    for value in &observation.fields {
        encode_value(bytes, *value);
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

fn encode_value(bytes: &mut Vec<u8>, value: PublicValue) {
    match value {
        PublicValue::Bit(value) => {
            bytes.push(1);
            bytes.push(u8::from(value));
        }
        PublicValue::Count(value) => {
            bytes.push(2);
            bytes.extend_from_slice(&value.to_le_bytes());
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
    use crate::benchmarks::eureka::consequence::{ConsequenceMetrics, PredictionOutcome};
    use crate::benchmarks::eureka::hidden_world::{
        CorpusPartition, EvaluatorWorld, FixtureFamily, WorldBuildProfile,
    };

    fn evaluator(seed: u64) -> EvaluatorWorld {
        EvaluatorWorld::build(WorldBuildProfile {
            family: FixtureFamily::CausalBits,
            seed,
            mechanism_variant: 0,
            partition: CorpusPartition::HeldOutEvaluation,
        })
    }

    fn prediction(action: PublicAction, fields: Vec<PublicValue>) -> ConsequencePrediction {
        ConsequencePrediction {
            action,
            outcome: PredictionOutcome::Predicted { fields },
        }
    }

    fn metrics(score: ConsequenceScore) -> ConsequenceMetrics {
        match score {
            ConsequenceScore::Scored(metrics) => metrics,
            other => panic!("expected score, got {other:?}"),
        }
    }

    #[test]
    fn frozen_prediction_is_immune_to_caller_mutation() {
        let mut world = evaluator(3);
        let pre = world.runtime().observe();
        let legal = world.runtime().legal_actions();
        let action = PublicAction::NoOp;
        let mut caller_prediction = prediction(action, pre.fields.clone());
        let mut custodian = PredictionCustodian::new(world.world_digest());
        let frozen = custodian.freeze(&pre, &legal, &caller_prediction).unwrap();
        let frozen_digest = frozen.commitment_digest();

        caller_prediction.outcome = PredictionOutcome::OutOfQualifiedDomain;
        assert_eq!(frozen.commitment_digest(), frozen_digest);

        let fresh = world.runtime().step(action);
        let trial = custodian.score_after_reveal(frozen, &fresh).unwrap();
        assert!(matches!(trial.candidate_score, ConsequenceScore::Scored(_)));
    }

    #[test]
    fn commit_sequence_precedes_reveal_sequence() {
        let mut world = evaluator(3);
        let pre = world.runtime().observe();
        let legal = world.runtime().legal_actions();
        let action = PublicAction::NoOp;
        let candidate = prediction(action, pre.fields.clone());
        let mut custodian = PredictionCustodian::new(world.world_digest());
        let frozen = custodian.freeze(&pre, &legal, &candidate).unwrap();
        let commit_sequence = frozen.commit_sequence();
        let fresh = world.runtime().step(action);
        let trial = custodian.score_after_reveal(frozen, &fresh).unwrap();
        assert!(trial.reveal_sequence > commit_sequence);
    }

    #[test]
    fn illegal_action_is_rejected_before_freeze() {
        let mut world = evaluator(3);
        let pre = world.runtime().observe();
        let legal = world.runtime().legal_actions();
        let candidate = prediction(
            PublicAction::Transfer {
                from: 0,
                to: 1,
                amount: 1,
            },
            pre.fields.clone(),
        );
        let mut custodian = PredictionCustodian::new(world.world_digest());
        assert_eq!(
            custodian.freeze(&pre, &legal, &candidate),
            Err(CustodyError::IllegalActionAtFreeze)
        );
    }

    #[test]
    fn reveal_action_must_match_frozen_prediction() {
        let mut world = evaluator(3);
        let pre = world.runtime().observe();
        let legal = world.runtime().legal_actions();
        let candidate = prediction(PublicAction::NoOp, pre.fields.clone());
        let mut custodian = PredictionCustodian::new(world.world_digest());
        let frozen = custodian.freeze(&pre, &legal, &candidate).unwrap();
        let fresh = world.runtime().step(PublicAction::Pulse { slot: 0 });
        assert_eq!(
            custodian.score_after_reveal(frozen, &fresh),
            Err(CustodyError::ActionMismatchAtReveal)
        );
    }

    #[test]
    fn frozen_prediction_cannot_cross_evaluator_worlds() {
        let mut world_a = evaluator(3);
        let mut world_b = evaluator(4);
        let pre = world_a.runtime().observe();
        let legal = world_a.runtime().legal_actions();
        let candidate = prediction(PublicAction::NoOp, pre.fields.clone());
        let mut custodian_a = PredictionCustodian::new(world_a.world_digest());
        let frozen = custodian_a.freeze(&pre, &legal, &candidate).unwrap();
        let fresh = world_b.runtime().step(PublicAction::NoOp);
        let mut custodian_b = PredictionCustodian::new(world_b.world_digest());
        assert_eq!(
            custodian_b.score_after_reveal(frozen, &fresh),
            Err(CustodyError::WrongEvaluatorWorld)
        );
    }

    #[test]
    fn stale_nonadvancing_outcome_is_rejected() {
        let mut world = evaluator(3);
        let pre = world.runtime().observe();
        let legal = world.runtime().legal_actions();
        let candidate = prediction(PublicAction::NoOp, pre.fields.clone());
        let mut custodian = PredictionCustodian::new(world.world_digest());
        let frozen = custodian.freeze(&pre, &legal, &candidate).unwrap();
        let stale = StepReceipt {
            action: PublicAction::NoOp,
            observation: pre.clone(),
        };
        assert_eq!(
            custodian.score_after_reveal(frozen, &stale),
            Err(CustodyError::NonAdvancingOutcome)
        );
    }

    #[test]
    fn candidate_and_copy_baseline_share_exact_transition() {
        let mut world = evaluator(3);
        let pre = world.runtime().observe();
        let legal = world.runtime().legal_actions();
        let action = PublicAction::NoOp;
        let candidate = prediction(action, pre.fields.clone());
        let mut custodian = PredictionCustodian::new(world.world_digest());
        let frozen = custodian.freeze(&pre, &legal, &candidate).unwrap();
        let fresh = world.runtime().step(action);
        let trial = custodian.score_after_reveal(frozen, &fresh).unwrap();

        assert_eq!(trial.action, action);
        assert_eq!(trial.baseline_id, COPY_CURRENT_STATE_BASELINE_ID);
        assert_ne!(trial.transition_digest, 0);
        let copy = metrics(trial.copy_baseline_score);
        let candidate = metrics(trial.candidate_score);
        assert_eq!(copy, candidate);
    }

    #[test]
    fn prediction_change_changes_commitment_identity() {
        let mut world = evaluator(3);
        let pre = world.runtime().observe();
        let legal = world.runtime().legal_actions();
        let mut fields_a = pre.fields.clone();
        let mut fields_b = pre.fields.clone();
        fields_a[0] = PublicValue::Bit(false);
        fields_b[0] = PublicValue::Bit(true);
        let mut custodian = PredictionCustodian::new(world.world_digest());
        let a = custodian
            .freeze(&pre, &legal, &prediction(PublicAction::NoOp, fields_a))
            .unwrap();
        let b = custodian
            .freeze(&pre, &legal, &prediction(PublicAction::NoOp, fields_b))
            .unwrap();
        assert_ne!(a.commitment_digest(), b.commitment_digest());
    }
}
