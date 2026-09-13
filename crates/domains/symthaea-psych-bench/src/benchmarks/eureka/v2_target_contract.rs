// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 V2 normalized production-FEP target adapter.
//!
//! This module freezes representation semantics only. It does not construct
//! evaluator worlds, choose schedules, train the target, or execute HeldOut.
//! Integer public benchmark semantics remain authoritative; the normalization is
//! a fixed, lossless-on-grid bridge into the production FEP's 4-D observation
//! interface.

use symthaea_fep::{ActionOutcome, FepHeldOutSubject, FrozenPredictionCommitment};

use super::consequence::{ConsequencePrediction, PredictionOutcome};
use super::hidden_world::{PublicAction, PublicValue};
use super::target_contract::EurekaTargetScope;
use super::v2_public_schema::{
    V2_CANONICAL_ACTIONS, V2_CONTEXT_DENOMINATOR, V2_COUNT_DENOMINATOR, V2_OBSERVATION_DIM,
    V2_REQUIRED_ACTIONS, V2PublicSchemaError, V2PublicState, action_index,
    public_schema_commitment as canonical_public_schema_commitment,
};

pub(super) const V2_FEP_TARGET_ADAPTER_REVISION: &str =
    "EUREKA.002.V2.PRODUCTION_FEP_NORMALIZED_ADAPTER.v2";
pub(super) const V2_FEP_TARGET_CONTRACT_REVISION: &str =
    "EUREKA.002.V2.PRODUCTION_FEP_TARGET_CONTRACT.v3";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2TargetContractError {
    PublicSchema(V2PublicSchemaError),
    ObservationDimensionMismatch { expected: usize, actual: usize },
    InsufficientActionCapacity { required: usize, actual: usize },
    OutcomeActionMismatch { expected: usize, actual: usize },
    TargetOutputDimensionMismatch { expected: usize, actual: usize },
    NonFiniteTargetOutput { index: usize },
    TargetValueOutOfI32Range { index: usize },
}

impl From<V2PublicSchemaError> for V2TargetContractError {
    fn from(value: V2PublicSchemaError) -> Self {
        Self::PublicSchema(value)
    }
}

/// Stateless shared adapter used identically for both V2 public world families.
pub(super) struct V2FepAdapter;

impl V2FepAdapter {
    /// Lossless-on-grid mapping into the production FEP observation scale.
    pub(super) fn encode_state(state: V2PublicState) -> [f64; V2_OBSERVATION_DIM] {
        let fields = state.fields();
        [
            f64::from(fields[0]) / f64::from(V2_COUNT_DENOMINATOR),
            f64::from(fields[1]) / f64::from(V2_COUNT_DENOMINATOR),
            f64::from(fields[2]) / f64::from(V2_COUNT_DENOMINATOR),
            f64::from(fields[3]) / f64::from(V2_CONTEXT_DENOMINATOR),
        ]
    }

    /// One exact action bijection shared by both V2 families.
    pub(super) fn encode_action(action: PublicAction) -> Result<usize, V2TargetContractError> {
        Ok(action_index(action)?)
    }

    /// Decode only public expected-observation semantics.
    ///
    /// No clipping/repair is performed. Finite out-of-domain predictions remain
    /// ordinary wrong integer predictions so the scorer can penalize them.
    pub(super) fn decode_expected_state(
        expected_observation: &[f64],
    ) -> Result<[i32; V2_OBSERVATION_DIM], V2TargetContractError> {
        if expected_observation.len() != V2_OBSERVATION_DIM {
            return Err(V2TargetContractError::TargetOutputDimensionMismatch {
                expected: V2_OBSERVATION_DIM,
                actual: expected_observation.len(),
            });
        }
        if let Some(index) = expected_observation
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(V2TargetContractError::NonFiniteTargetOutput { index });
        }

        let scales = [
            f64::from(V2_COUNT_DENOMINATOR),
            f64::from(V2_COUNT_DENOMINATOR),
            f64::from(V2_COUNT_DENOMINATOR),
            f64::from(V2_CONTEXT_DENOMINATOR),
        ];
        let mut output = [0_i32; V2_OBSERVATION_DIM];
        for index in 0..V2_OBSERVATION_DIM {
            let scaled = expected_observation[index] * scales[index];
            if !scaled.is_finite()
                || scaled < f64::from(i32::MIN)
                || scaled > f64::from(i32::MAX)
            {
                return Err(V2TargetContractError::TargetValueOutOfI32Range { index });
            }
            output[index] = scaled.round() as i32;
        }
        Ok(output)
    }

    /// Convert one typed FEP action outcome to the canonical EUREKA consequence
    /// prediction. The internal 8-D hidden-state prediction is intentionally
    /// ignored; only `expected_observation` has public consequence semantics.
    pub(super) fn prediction_from_outcome(
        action: PublicAction,
        outcome: &ActionOutcome,
    ) -> Result<ConsequencePrediction, V2TargetContractError> {
        let expected_action = Self::encode_action(action)?;
        if outcome.action != expected_action {
            return Err(V2TargetContractError::OutcomeActionMismatch {
                expected: expected_action,
                actual: outcome.action,
            });
        }
        let decoded = Self::decode_expected_state(&outcome.expected_observation)?;
        Ok(ConsequencePrediction {
            action,
            outcome: PredictionOutcome::Predicted {
                fields: decoded.into_iter().map(PublicValue::Count).collect(),
            },
        })
    }
}

/// Cryptographically bound held-out target contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2FepTargetContract {
    scope: EurekaTargetScope,
    public_schema_commitment: [u8; 32],
    learned_subject_commitment: FrozenPredictionCommitment,
    observation_dim: usize,
    action_count: usize,
    commitment: [u8; 32],
}

impl V2FepTargetContract {
    /// Mint the target contract from the narrowed runner-facing authority, not a
    /// raw trainable snapshot/session.
    pub(super) fn from_heldout_subject(
        subject: &FepHeldOutSubject,
    ) -> Result<Self, V2TargetContractError> {
        if subject.observation_dim() != V2_OBSERVATION_DIM {
            return Err(V2TargetContractError::ObservationDimensionMismatch {
                expected: V2_OBSERVATION_DIM,
                actual: subject.observation_dim(),
            });
        }
        if subject.action_count() < V2_REQUIRED_ACTIONS {
            return Err(V2TargetContractError::InsufficientActionCapacity {
                required: V2_REQUIRED_ACTIONS,
                actual: subject.action_count(),
            });
        }
        let public_schema_commitment = canonical_public_schema_commitment();
        let learned_subject_commitment = subject.commitment();
        let commitment = contract_commitment(
            EurekaTargetScope::ProductionFepComponentSnapshot,
            public_schema_commitment,
            learned_subject_commitment,
            subject.observation_dim(),
            subject.action_count(),
        );
        Ok(Self {
            scope: EurekaTargetScope::ProductionFepComponentSnapshot,
            public_schema_commitment,
            learned_subject_commitment,
            observation_dim: subject.observation_dim(),
            action_count: subject.action_count(),
            commitment,
        })
    }

    pub(super) const fn scope(self) -> EurekaTargetScope {
        self.scope
    }

    pub(super) const fn public_schema_commitment(self) -> [u8; 32] {
        self.public_schema_commitment
    }

    pub(super) const fn learned_subject_commitment(self) -> FrozenPredictionCommitment {
        self.learned_subject_commitment
    }

    pub(super) const fn observation_dim(self) -> usize {
        self.observation_dim
    }

    pub(super) const fn action_count(self) -> usize {
        self.action_count
    }

    pub(super) const fn commitment(self) -> [u8; 32] {
        self.commitment
    }
}

fn contract_commitment(
    scope: EurekaTargetScope,
    public_schema_commitment: [u8; 32],
    learned_subject_commitment: FrozenPredictionCommitment,
    observation_dim: usize,
    action_count: usize,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_FEP_TARGET_CONTRACT_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_FEP_TARGET_ADAPTER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment);
    bytes.push(match scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    bytes.extend_from_slice(&(observation_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&(action_count as u64).to_le_bytes());
    bytes.extend_from_slice(learned_subject_commitment.as_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fep::{
        ActiveInferenceAgent, ActiveInferenceAgentConfig, FepPredictionSession, HiddenState,
    };

    fn heldout_subject() -> FepHeldOutSubject {
        let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 4,
            enable_model_learning: true,
            enable_td_learning: true,
            ..Default::default()
        });
        let snapshot = FepPredictionSession::from_agent(&agent)
            .freeze_for_evaluation()
            .unwrap();
        FepHeldOutSubject::seal(snapshot)
    }

    #[test]
    fn every_legal_public_channel_value_round_trips_exactly() {
        for count in 0..=V2_COUNT_DENOMINATOR {
            let state = V2PublicState::new([count, count, count, 0]).unwrap();
            assert_eq!(
                V2FepAdapter::decode_expected_state(&V2FepAdapter::encode_state(state)).unwrap(),
                state.fields()
            );
        }
        for context in 0..=V2_CONTEXT_DENOMINATOR {
            let state = V2PublicState::new([0, 1, 31, context]).unwrap();
            assert_eq!(
                V2FepAdapter::decode_expected_state(&V2FepAdapter::encode_state(state)).unwrap(),
                state.fields()
            );
        }
    }

    #[test]
    fn representative_cartesian_states_round_trip() {
        for a in [0, 1, 15, 31] {
            for b in [0, 7, 16, 31] {
                for c in [0, 11, 23, 31] {
                    for context in 0..=V2_CONTEXT_DENOMINATOR {
                        let state = V2PublicState::new([a, b, c, context]).unwrap();
                        assert_eq!(
                            V2FepAdapter::decode_expected_state(&V2FepAdapter::encode_state(state))
                                .unwrap(),
                            state.fields()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn action_mapping_consumes_the_canonical_public_schema() {
        for (index, action) in V2_CANONICAL_ACTIONS.into_iter().enumerate() {
            assert_eq!(V2FepAdapter::encode_action(action), Ok(index));
        }
        assert_eq!(
            V2FepAdapter::encode_action(PublicAction::Pulse { slot: 3 }),
            Err(V2TargetContractError::PublicSchema(
                V2PublicSchemaError::UnsupportedAction
            ))
        );
        assert!(matches!(
            V2FepAdapter::encode_action(PublicAction::Transfer {
                from: 0,
                to: 1,
                amount: 1
            }),
            Err(V2TargetContractError::PublicSchema(
                V2PublicSchemaError::UnsupportedAction
            ))
        ));
    }

    #[test]
    fn decoder_does_not_clip_finite_wrong_predictions() {
        let decoded = V2FepAdapter::decode_expected_state(&[1.2, -0.2, 0.5, 1.4]).unwrap();
        assert_eq!(decoded[0], 37);
        assert_eq!(decoded[1], -6);
        assert_eq!(decoded[2], 16);
        assert_eq!(decoded[3], 10);
    }

    #[test]
    fn nonfinite_and_unsafe_outputs_fail_explicitly() {
        assert_eq!(
            V2FepAdapter::decode_expected_state(&[0.0, f64::NAN, 0.0, 0.0]),
            Err(V2TargetContractError::NonFiniteTargetOutput { index: 1 })
        );
        assert_eq!(
            V2FepAdapter::decode_expected_state(&[f64::MAX, 0.0, 0.0, 0.0]),
            Err(V2TargetContractError::TargetValueOutOfI32Range { index: 0 })
        );
    }

    #[test]
    fn consequence_decoder_uses_expected_observation_not_hidden_state() {
        let mut hidden_a = HiddenState::new(8);
        let mut hidden_b = HiddenState::new(8);
        hidden_a.mean[0] = 0.1;
        hidden_b.mean[0] = 0.9;
        let expected = vec![1.0 / 31.0, 2.0 / 31.0, 3.0 / 31.0, 4.0 / 7.0];
        let a = ActionOutcome {
            action: 2,
            predicted_next_state: hidden_a,
            expected_observation: expected.clone(),
            timestamp: 1,
        };
        let b = ActionOutcome {
            action: 2,
            predicted_next_state: hidden_b,
            expected_observation: expected,
            timestamp: 999,
        };
        let public_action = PublicAction::Pulse { slot: 1 };
        assert_eq!(
            V2FepAdapter::prediction_from_outcome(public_action, &a).unwrap(),
            V2FepAdapter::prediction_from_outcome(public_action, &b).unwrap()
        );
    }

    #[test]
    fn outcome_action_mismatch_fails_closed() {
        let outcome = ActionOutcome {
            action: 0,
            predicted_next_state: HiddenState::new(8),
            expected_observation: vec![0.0; V2_OBSERVATION_DIM],
            timestamp: 0,
        };
        assert_eq!(
            V2FepAdapter::prediction_from_outcome(PublicAction::Pulse { slot: 0 }, &outcome),
            Err(V2TargetContractError::OutcomeActionMismatch {
                expected: 1,
                actual: 0
            })
        );
    }

    #[test]
    fn contract_is_component_scoped_and_binds_schema_and_learned_commitment() {
        let subject = heldout_subject();
        let learned = subject.commitment();
        let schema = canonical_public_schema_commitment();
        let contract = V2FepTargetContract::from_heldout_subject(&subject).unwrap();
        assert_eq!(
            contract.scope(),
            EurekaTargetScope::ProductionFepComponentSnapshot
        );
        assert_eq!(contract.observation_dim(), V2_OBSERVATION_DIM);
        assert_eq!(contract.action_count(), V2_REQUIRED_ACTIONS);
        assert_eq!(contract.public_schema_commitment(), schema);
        assert_eq!(contract.learned_subject_commitment(), learned);
        assert_ne!(contract.commitment(), [0_u8; 32]);
    }

    #[test]
    fn contract_commitment_changes_with_learned_model() {
        let a = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 4,
            ..Default::default()
        });
        let mut b = a.clone();
        b.model.transition_matrices[0][0][0] += 0.01;
        let subject_a = FepHeldOutSubject::seal(
            FepPredictionSession::from_agent(&a)
                .freeze_for_evaluation()
                .unwrap(),
        );
        let subject_b = FepHeldOutSubject::seal(
            FepPredictionSession::from_agent(&b)
                .freeze_for_evaluation()
                .unwrap(),
        );
        let contract_a = V2FepTargetContract::from_heldout_subject(&subject_a).unwrap();
        let contract_b = V2FepTargetContract::from_heldout_subject(&subject_b).unwrap();
        assert_eq!(
            contract_a.public_schema_commitment(),
            contract_b.public_schema_commitment()
        );
        assert_ne!(
            contract_a.learned_subject_commitment(),
            contract_b.learned_subject_commitment()
        );
        assert_ne!(contract_a.commitment(), contract_b.commitment());
    }
}
