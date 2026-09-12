// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Target-scope and lossless-adapter contract for production FEP evidence.
//!
//! This module does not execute held-out campaigns. It freezes the semantics a
//! future EUREKA production-FEP adapter must satisfy before prediction begins.

use symthaea_fep::FepEvaluationSnapshot;

use super::hidden_world::{FixtureFamily, PublicAction, PublicObservation, PublicValue};

pub const FEP_TARGET_ADAPTER_REVISION: &str = "EUREKA.TARGET.PRODUCTION_FEP.v1";

/// The architectural scope to which an EUREKA target result applies.
///
/// Component evidence must never be silently relabeled as whole-loop evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EurekaTargetScope {
    ProductionFepComponentSnapshot,
    FullCognitiveLoop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FepTargetContractError {
    InsufficientObservationCapacity { required: usize, available: usize },
    InsufficientActionCapacity { required: usize, available: usize },
    PublicFieldCountMismatch { expected: usize, actual: usize },
    PublicFieldTypeMismatch { index: usize },
    UnsupportedPublicAction,
    TargetOutputDimensionMismatch { expected: usize, actual: usize },
    NonFiniteTargetOutput { index: usize },
    TargetCountOutOfRange { index: usize },
}

/// Frozen mapping between one EUREKA fixture family and a learned production
/// FEP evaluation snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FepTargetContract {
    scope: EurekaTargetScope,
    family: FixtureFamily,
    snapshot_replay_digest: u64,
    observation_dim: usize,
    action_count: usize,
    replay_digest: u64,
}

impl FepTargetContract {
    pub(super) fn new(
        snapshot: &FepEvaluationSnapshot,
        family: FixtureFamily,
    ) -> Result<Self, FepTargetContractError> {
        let required_observations = required_observation_dim(family);
        let required_actions = required_action_count(family);
        if snapshot.observation_dim() < required_observations {
            return Err(FepTargetContractError::InsufficientObservationCapacity {
                required: required_observations,
                available: snapshot.observation_dim(),
            });
        }
        if snapshot.action_count() < required_actions {
            return Err(FepTargetContractError::InsufficientActionCapacity {
                required: required_actions,
                available: snapshot.action_count(),
            });
        }

        let mut contract = Self {
            scope: EurekaTargetScope::ProductionFepComponentSnapshot,
            family,
            snapshot_replay_digest: snapshot.replay_digest(),
            observation_dim: snapshot.observation_dim(),
            action_count: snapshot.action_count(),
            replay_digest: 0,
        };
        contract.replay_digest = contract_digest(&contract);
        Ok(contract)
    }

    pub(super) fn scope(&self) -> EurekaTargetScope {
        self.scope
    }

    pub(super) fn family(&self) -> FixtureFamily {
        self.family
    }

    pub(super) fn snapshot_replay_digest(&self) -> u64 {
        self.snapshot_replay_digest
    }

    pub(super) fn replay_digest(&self) -> u64 {
        self.replay_digest
    }

    pub(super) fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    pub(super) fn action_count(&self) -> usize {
        self.action_count
    }

    /// Encode only target-visible public state. Hidden role/permutation/mechanism
    /// metadata is neither accepted nor representable by this function.
    ///
    /// V1 channel semantics:
    ///
    /// - ResourceFlow: `[stock0, stock1, stock2, step]`
    /// - CausalBits: `[bit0, bit1, bit2, bit3, step]`
    ///
    /// Any extra target observation channels are filled with zero and are not
    /// treated as evidence-bearing public state.
    pub(super) fn encode_observation(
        &self,
        observation: &PublicObservation,
    ) -> Result<Vec<f64>, FepTargetContractError> {
        let expected_fields = public_field_count(self.family);
        if observation.fields.len() != expected_fields {
            return Err(FepTargetContractError::PublicFieldCountMismatch {
                expected: expected_fields,
                actual: observation.fields.len(),
            });
        }

        let mut encoded = vec![0.0; self.observation_dim];
        match self.family {
            FixtureFamily::ResourceFlow => {
                for (index, value) in observation.fields.iter().enumerate() {
                    let PublicValue::Count(count) = value else {
                        return Err(FepTargetContractError::PublicFieldTypeMismatch { index });
                    };
                    encoded[index] = f64::from(*count);
                }
                encoded[3] = f64::from(observation.step);
            }
            FixtureFamily::CausalBits => {
                for (index, value) in observation.fields.iter().enumerate() {
                    let PublicValue::Bit(bit) = value else {
                        return Err(FepTargetContractError::PublicFieldTypeMismatch { index });
                    };
                    encoded[index] = if *bit { 1.0 } else { 0.0 };
                }
                encoded[4] = f64::from(observation.step);
            }
        }
        Ok(encoded)
    }

    /// Deterministic bijection over the fixture family's canonical legal action
    /// vocabulary. Unsupported/non-canonical requests fail rather than alias.
    pub(super) fn encode_action(
        &self,
        action: PublicAction,
    ) -> Result<usize, FepTargetContractError> {
        let target = match (self.family, action) {
            (FixtureFamily::ResourceFlow, PublicAction::NoOp) => 0,
            (
                FixtureFamily::ResourceFlow,
                PublicAction::Transfer {
                    from: 0,
                    to: 1,
                    amount: 1,
                },
            ) => 1,
            (
                FixtureFamily::ResourceFlow,
                PublicAction::Transfer {
                    from: 1,
                    to: 2,
                    amount: 1,
                },
            ) => 2,
            (FixtureFamily::CausalBits, PublicAction::NoOp) => 0,
            (FixtureFamily::CausalBits, PublicAction::Pulse { slot: 0 }) => 1,
            (FixtureFamily::CausalBits, PublicAction::Pulse { slot: 1 }) => 2,
            (FixtureFamily::CausalBits, PublicAction::Pulse { slot: 2 }) => 3,
            (FixtureFamily::CausalBits, PublicAction::Pulse { slot: 3 }) => 4,
            _ => return Err(FepTargetContractError::UnsupportedPublicAction),
        };
        if target >= self.action_count {
            return Err(FepTargetContractError::InsufficientActionCapacity {
                required: target + 1,
                available: self.action_count,
            });
        }
        Ok(target)
    }

    /// Decode the target's typed expected-observation vector into the public
    /// field semantics scored by EUREKA. The public step channel is deliberately
    /// not scored as a consequence field in EUREKA-002.
    pub(super) fn decode_expected_fields(
        &self,
        expected_observation: &[f64],
    ) -> Result<Vec<PublicValue>, FepTargetContractError> {
        if expected_observation.len() != self.observation_dim {
            return Err(FepTargetContractError::TargetOutputDimensionMismatch {
                expected: self.observation_dim,
                actual: expected_observation.len(),
            });
        }
        if let Some(index) = expected_observation
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(FepTargetContractError::NonFiniteTargetOutput { index });
        }

        match self.family {
            FixtureFamily::CausalBits => Ok(expected_observation[..4]
                .iter()
                .map(|value| PublicValue::Bit(*value >= 0.5))
                .collect()),
            FixtureFamily::ResourceFlow => expected_observation[..3]
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let rounded = value.round();
                    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
                        return Err(FepTargetContractError::TargetCountOutOfRange { index });
                    }
                    Ok(PublicValue::Count(rounded as i32))
                })
                .collect(),
        }
    }
}

pub(super) const fn required_observation_dim(family: FixtureFamily) -> usize {
    match family {
        FixtureFamily::CausalBits => 5,   // four public bits + public step
        FixtureFamily::ResourceFlow => 4, // three public stocks + public step
    }
}

pub(super) const fn required_action_count(family: FixtureFamily) -> usize {
    match family {
        FixtureFamily::CausalBits => 5,   // NoOp + Pulse(0..3)
        FixtureFamily::ResourceFlow => 3, // NoOp + two canonical transfers
    }
}

const fn public_field_count(family: FixtureFamily) -> usize {
    match family {
        FixtureFamily::CausalBits => 4,
        FixtureFamily::ResourceFlow => 3,
    }
}

fn contract_digest(contract: &FepTargetContract) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(FEP_TARGET_ADAPTER_REVISION.as_bytes());
    bytes.push(0);
    bytes.push(match contract.scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    bytes.push(match contract.family {
        FixtureFamily::CausalBits => 1,
        FixtureFamily::ResourceFlow => 2,
    });
    bytes.extend_from_slice(&contract.snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&(contract.observation_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&(contract.action_count as u64).to_le_bytes());
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
    use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, FepPredictionSession};

    fn snapshot(obs_dim: usize, action_count: usize) -> FepEvaluationSnapshot {
        let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim,
            num_actions: action_count,
            enable_td_learning: true,
            ..Default::default()
        });
        FepPredictionSession::from_agent(&agent)
            .freeze_for_evaluation()
            .unwrap()
    }

    #[test]
    fn current_cognitive_loop_dimensions_can_represent_resource_flow() {
        let frozen = snapshot(4, 4);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        assert_eq!(contract.scope(), EurekaTargetScope::ProductionFepComponentSnapshot);
        assert_eq!(contract.observation_dim(), 4);
        assert_eq!(contract.action_count(), 4);
    }

    #[test]
    fn current_cognitive_loop_dimensions_reject_causal_bits() {
        let frozen = snapshot(4, 4);
        assert!(matches!(
            FepTargetContract::new(&frozen, FixtureFamily::CausalBits),
            Err(FepTargetContractError::InsufficientObservationCapacity {
                required: 5,
                available: 4
            })
        ));
    }

    #[test]
    fn causal_bits_also_requires_five_distinct_actions() {
        let frozen = snapshot(5, 4);
        assert!(matches!(
            FepTargetContract::new(&frozen, FixtureFamily::CausalBits),
            Err(FepTargetContractError::InsufficientActionCapacity {
                required: 5,
                available: 4
            })
        ));
    }

    #[test]
    fn resource_flow_observation_preserves_all_public_state() {
        let frozen = snapshot(4, 4);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        let observation = PublicObservation {
            step: 7,
            fields: vec![
                PublicValue::Count(2),
                PublicValue::Count(5),
                PublicValue::Count(9),
            ],
        };
        assert_eq!(
            contract.encode_observation(&observation).unwrap(),
            vec![2.0, 5.0, 9.0, 7.0]
        );
    }

    #[test]
    fn resource_flow_action_map_is_bijective_over_canonical_vocabulary() {
        let frozen = snapshot(4, 4);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        let actions = [
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
        ];
        let mapped: Vec<usize> = actions
            .iter()
            .map(|action| contract.encode_action(*action).unwrap())
            .collect();
        assert_eq!(mapped, vec![0, 1, 2]);
        let mut unique = mapped.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), actions.len());
    }

    #[test]
    fn noncanonical_action_never_aliases_to_valid_target_action() {
        let frozen = snapshot(4, 4);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        assert_eq!(
            contract.encode_action(PublicAction::Transfer {
                from: 0,
                to: 2,
                amount: 1,
            }),
            Err(FepTargetContractError::UnsupportedPublicAction)
        );
    }

    #[test]
    fn deterministic_decoder_has_explicit_bit_and_count_semantics() {
        let resource = FepTargetContract::new(&snapshot(4, 4), FixtureFamily::ResourceFlow).unwrap();
        assert_eq!(
            resource
                .decode_expected_fields(&[1.49, 2.51, -0.2, 8.9])
                .unwrap(),
            vec![
                PublicValue::Count(1),
                PublicValue::Count(3),
                PublicValue::Count(0),
            ]
        );

        let bits = FepTargetContract::new(&snapshot(5, 5), FixtureFamily::CausalBits).unwrap();
        assert_eq!(
            bits.decode_expected_fields(&[0.49, 0.5, 0.9, -1.0, 4.0])
                .unwrap(),
            vec![
                PublicValue::Bit(false),
                PublicValue::Bit(true),
                PublicValue::Bit(true),
                PublicValue::Bit(false),
            ]
        );
    }

    #[test]
    fn contract_identity_binds_frozen_model_and_family() {
        let a = snapshot(5, 5);
        let b = snapshot(5, 5);
        let bits_a = FepTargetContract::new(&a, FixtureFamily::CausalBits).unwrap();
        let bits_b = FepTargetContract::new(&b, FixtureFamily::CausalBits).unwrap();
        let flow = FepTargetContract::new(&a, FixtureFamily::ResourceFlow).unwrap();
        assert_eq!(bits_a.replay_digest(), bits_b.replay_digest());
        assert_ne!(bits_a.replay_digest(), flow.replay_digest());
        assert_eq!(bits_a.snapshot_replay_digest(), a.replay_digest());
    }
}