// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen qualification control surface for `AdaptiveReasoner`.
//!
//! This module deliberately separates mechanism execution from learning and evaluation.
//! A qualification run borrows the reasoner immutably, injects a deterministic RNG seed,
//! optionally forces preregistered actions at exact steps, executes the exact declared step
//! budget, and emits a content-bound receipt. It does not update the Q-table or replay buffer,
//! does not consume an expected answer, and does not convert local HDC transition distance into
//! task reward.

use super::{
    get_standard_primitives, AdaptiveReasoner, Experience, QLearningAgent, ReasoningAction,
    ReasoningState,
};
use crate::consciousness::primitive_reasoning::{ReasoningChain, TransformationType};
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::Primitive;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

pub const ADAPTIVE_QUALIFICATION_CONTROL_VERSION: &str = "rq-006z-adaptive-control-v1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea/reasoning/adaptive-qualification-receipt/v1";
const LEARNER_STATE_DOMAIN: &[u8] = b"symthaea/reasoning/adaptive-learner-state/v1";
const REASONING_STATE_DOMAIN: &[u8] = b"symthaea/reasoning/adaptive-reasoning-state/v1";
const EXECUTION_DOMAIN: &[u8] = b"symthaea/reasoning/adaptive-qualification-execution/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationActionSource {
    Policy,
    Forced,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForcedAdaptiveAction {
    pub step: usize,
    pub primitive_name: String,
    pub transformation: TransformationType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveQualificationConfig {
    /// Exact action budget. Qualification never exits early on transition-distance heuristics.
    pub max_steps: usize,
    /// Deterministic RNG lineage for policy exploration.
    pub rng_seed: u64,
    /// Optional preregistered interventions. At most one action may be forced at each step.
    pub forced_actions: Vec<ForcedAdaptiveAction>,
}

impl AdaptiveQualificationConfig {
    pub fn frozen(max_steps: usize, rng_seed: u64) -> Self {
        Self {
            max_steps,
            rng_seed,
            forced_actions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveQualificationActionRecord {
    pub step: usize,
    pub source: QualificationActionSource,
    /// Action selected by the frozen policy before any forced intervention is applied.
    pub policy_action: ReasoningAction,
    /// Action actually executed.
    pub executed_action: ReasoningAction,
    pub executed_primitive_encoding_digest: String,
    pub pre_state_digest: String,
    pub post_state_digest: String,
    /// Legacy production field retained only as transition-geometry provenance.
    pub local_transition_contribution: f64,
    pub execution_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveQualificationReceipt {
    pub control_version: String,
    pub rng_seed: u64,
    pub max_steps: usize,
    pub question_digest: String,
    pub learner_state_before: String,
    pub learner_state_after: String,
    pub final_state_digest: String,
    pub forced_actions_requested: usize,
    pub forced_actions_applied: usize,
    pub actions: Vec<AdaptiveQualificationActionRecord>,
    pub receipt_commitment: String,
}

impl AdaptiveQualificationReceipt {
    pub fn learner_state_unchanged(&self) -> bool {
        self.learner_state_before == self.learner_state_after
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveQualificationRun {
    pub chain: ReasoningChain,
    pub receipt: AdaptiveQualificationReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveQualificationError {
    ZeroStepBudget,
    ForcedStepOutsideBudget { step: usize, max_steps: usize },
    DuplicateForcedStep { step: usize },
    UnknownForcedPrimitive { step: usize, primitive_name: String },
    UnsupportedForcedTransformation {
        step: usize,
        transformation: TransformationType,
    },
    MissingExecution { step: usize },
    LearnerStateChanged,
}

impl fmt::Display for AdaptiveQualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStepBudget => write!(f, "adaptive qualification requires max_steps > 0"),
            Self::ForcedStepOutsideBudget { step, max_steps } => write!(
                f,
                "forced action step {step} lies outside qualification budget 0..{max_steps}"
            ),
            Self::DuplicateForcedStep { step } => {
                write!(f, "multiple forced actions target qualification step {step}")
            }
            Self::UnknownForcedPrimitive {
                step,
                primitive_name,
            } => write!(
                f,
                "forced action at step {step} references unknown adaptive primitive `{primitive_name}`"
            ),
            Self::UnsupportedForcedTransformation {
                step,
                transformation,
            } => write!(
                f,
                "forced action at step {step} uses transformation {transformation:?} outside the adaptive action space"
            ),
            Self::MissingExecution { step } => write!(
                f,
                "production reasoning chain emitted no execution for qualification step {step}"
            ),
            Self::LearnerStateChanged => write!(
                f,
                "frozen adaptive qualification changed learner state despite immutable execution"
            ),
        }
    }
}

impl std::error::Error for AdaptiveQualificationError {}

impl AdaptiveReasoner {
    /// Execute an exact-budget, deterministic, learning-frozen reasoning trial.
    ///
    /// The immutable receiver is intentional: Q-table, replay buffer, epsilon, and other learner
    /// state cannot be updated by this path. External task targets/rewards are not accepted here;
    /// they belong to a separate evaluator after solver-visible execution is complete.
    pub fn reason_adaptive_qualification_frozen(
        &self,
        question: BinaryHV,
        config: &AdaptiveQualificationConfig,
    ) -> Result<AdaptiveQualificationRun, AdaptiveQualificationError> {
        let primitives = get_standard_primitives();
        validate_config(config, &primitives)?;
        let available_actions = adaptive_action_space(&primitives);
        let learner_state_before = learner_state_commitment(self);
        let question_digest = binary_hv_digest(&question);
        let mut rng = StdRng::seed_from_u64(config.rng_seed);
        let mut chain = ReasoningChain::new(question);
        let mut records = Vec::with_capacity(config.max_steps);
        let mut forced_actions_applied = 0usize;

        for step in 0..config.max_steps {
            let pre_state = chain.current_state;
            let current_state = ReasoningState::from_chain(&chain);

            // Always execute the frozen policy selection first. A forced intervention overrides
            // the action only after policy RNG consumption, preserving matched RNG lineage.
            let policy = if self.use_rl {
                select_action_with_rng(&self.agent, &current_state, &available_actions, &mut rng)
            } else {
                available_actions
                    .first()
                    .cloned()
                    .unwrap_or_else(|| (primitives[0].clone(), TransformationType::Bind))
            };

            let policy_action = ReasoningAction::new(&policy.0, policy.1);
            let forced = config.forced_actions.iter().find(|action| action.step == step);
            let (executed_primitive, executed_transformation, source) = if let Some(forced) = forced {
                let primitive = primitives
                    .iter()
                    .find(|primitive| primitive.name == forced.primitive_name)
                    .expect("validated forced primitive must exist")
                    .clone();
                forced_actions_applied = forced_actions_applied.saturating_add(1);
                (
                    primitive,
                    forced.transformation,
                    QualificationActionSource::Forced,
                )
            } else {
                (policy.0, policy.1, QualificationActionSource::Policy)
            };

            chain
                .execute_primitive(&executed_primitive, executed_transformation)
                .map_err(|_| AdaptiveQualificationError::MissingExecution { step })?;
            let execution = chain
                .executions
                .last()
                .ok_or(AdaptiveQualificationError::MissingExecution { step })?;
            let post_state = chain.current_state;
            let executed_action = ReasoningAction::new(&executed_primitive, executed_transformation);
            let encoding_digest = binary_hv_digest(&executed_primitive.encoding);
            let pre_state_digest = binary_hv_digest(&pre_state);
            let post_state_digest = binary_hv_digest(&post_state);
            let execution_commitment = execution_commitment(
                step,
                source,
                &policy_action,
                &executed_action,
                &encoding_digest,
                &pre_state_digest,
                &post_state_digest,
                execution.phi_contribution,
            );

            records.push(AdaptiveQualificationActionRecord {
                step,
                source,
                policy_action,
                executed_action,
                executed_primitive_encoding_digest: encoding_digest,
                pre_state_digest,
                post_state_digest,
                local_transition_contribution: execution.phi_contribution,
                execution_commitment,
            });
        }

        let learner_state_after = learner_state_commitment(self);
        if learner_state_before != learner_state_after {
            return Err(AdaptiveQualificationError::LearnerStateChanged);
        }

        let final_state_digest = binary_hv_digest(&chain.current_state);
        let mut receipt = AdaptiveQualificationReceipt {
            control_version: ADAPTIVE_QUALIFICATION_CONTROL_VERSION.into(),
            rng_seed: config.rng_seed,
            max_steps: config.max_steps,
            question_digest,
            learner_state_before,
            learner_state_after,
            final_state_digest,
            forced_actions_requested: config.forced_actions.len(),
            forced_actions_applied,
            actions: records,
            receipt_commitment: String::new(),
        };
        receipt.receipt_commitment = receipt_commitment(&receipt);

        Ok(AdaptiveQualificationRun { chain, receipt })
    }

    /// Content-bound commitment to all mutable learner state relevant to adaptive policy/learning.
    pub fn adaptive_qualification_learner_state_commitment(&self) -> String {
        learner_state_commitment(self)
    }
}

fn adaptive_action_space(primitives: &[Primitive]) -> Vec<(Primitive, TransformationType)> {
    const TRANSFORMATIONS: [TransformationType; 4] = [
        TransformationType::Bind,
        TransformationType::Bundle,
        TransformationType::Resonate,
        TransformationType::Abstract,
    ];
    let mut actions = Vec::with_capacity(primitives.len() * TRANSFORMATIONS.len());
    for primitive in primitives {
        for transformation in TRANSFORMATIONS {
            actions.push((primitive.clone(), transformation));
        }
    }
    actions
}

fn validate_config(
    config: &AdaptiveQualificationConfig,
    primitives: &[Primitive],
) -> Result<(), AdaptiveQualificationError> {
    if config.max_steps == 0 {
        return Err(AdaptiveQualificationError::ZeroStepBudget);
    }
    let names = primitives
        .iter()
        .map(|primitive| primitive.name.as_str())
        .collect::<HashSet<_>>();
    let mut seen_steps = HashSet::new();
    for forced in &config.forced_actions {
        if forced.step >= config.max_steps {
            return Err(AdaptiveQualificationError::ForcedStepOutsideBudget {
                step: forced.step,
                max_steps: config.max_steps,
            });
        }
        if !seen_steps.insert(forced.step) {
            return Err(AdaptiveQualificationError::DuplicateForcedStep { step: forced.step });
        }
        if !names.contains(forced.primitive_name.as_str()) {
            return Err(AdaptiveQualificationError::UnknownForcedPrimitive {
                step: forced.step,
                primitive_name: forced.primitive_name.clone(),
            });
        }
        if !matches!(
            forced.transformation,
            TransformationType::Bind
                | TransformationType::Bundle
                | TransformationType::Resonate
                | TransformationType::Abstract
        ) {
            return Err(AdaptiveQualificationError::UnsupportedForcedTransformation {
                step: forced.step,
                transformation: forced.transformation,
            });
        }
    }
    Ok(())
}

fn select_action_with_rng<R: Rng + ?Sized>(
    agent: &QLearningAgent,
    state: &ReasoningState,
    available_actions: &[(Primitive, TransformationType)],
    rng: &mut R,
) -> (Primitive, TransformationType) {
    if available_actions.is_empty() {
        return (
            get_standard_primitives().remove(0),
            TransformationType::Bind,
        );
    }
    if rng.r#gen::<f64>() < agent.epsilon {
        let idx = rng.gen_range(0..available_actions.len());
        return available_actions[idx].clone();
    }

    let state_hash = state.state_hash();
    let mut best_q = f64::NEG_INFINITY;
    let mut best_action = available_actions[0].clone();
    for (primitive, transformation) in available_actions {
        let action = ReasoningAction::new(primitive, *transformation);
        let q = agent.get_q(state_hash, action.action_hash());
        if q > best_q {
            best_q = q;
            best_action = (primitive.clone(), *transformation);
        }
    }
    best_action
}

fn learner_state_commitment(reasoner: &AdaptiveReasoner) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, LEARNER_STATE_DOMAIN);
    hash_bool(&mut hasher, reasoner.use_rl);
    hash_u64(&mut hasher, reasoner.agent.alpha.to_bits());
    hash_u64(&mut hasher, reasoner.agent.gamma.to_bits());
    hash_u64(&mut hasher, reasoner.agent.epsilon.to_bits());
    hash_u64(&mut hasher, reasoner.agent.epsilon_decay.to_bits());
    hash_u64(&mut hasher, reasoner.agent.epsilon_min.to_bits());
    hash_u64(&mut hasher, reasoner.agent.max_buffer_size as u64);
    hash_u64(&mut hasher, reasoner.agent.max_q_table_size as u64);

    let mut q_entries = reasoner
        .agent
        .q_table
        .iter()
        .map(|(&(state, action), &value)| (state, action, value.to_bits()))
        .collect::<Vec<_>>();
    q_entries.sort_unstable_by_key(|entry| (entry.0, entry.1));
    hash_u64(&mut hasher, q_entries.len() as u64);
    for (state, action, value_bits) in q_entries {
        hash_u64(&mut hasher, state);
        hash_u64(&mut hasher, action);
        hash_u64(&mut hasher, value_bits);
    }

    hash_u64(&mut hasher, reasoner.agent.replay_buffer.len() as u64);
    for experience in &reasoner.agent.replay_buffer {
        hash_experience(&mut hasher, experience);
    }
    hasher.finalize().to_hex().to_string()
}

fn hash_experience(hasher: &mut blake3::Hasher, experience: &Experience) {
    hash_str(hasher, &reasoning_state_commitment(&experience.state));
    hash_str(hasher, &experience.action.primitive_name);
    hash_u64(hasher, transformation_tag(experience.action.transformation));
    hash_u64(hasher, experience.reward.to_bits());
    hash_str(hasher, &reasoning_state_commitment(&experience.next_state));
    hash_bool(hasher, experience.done);
}

fn reasoning_state_commitment(state: &ReasoningState) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, REASONING_STATE_DOMAIN);
    hash_u64(&mut hasher, state.answer_features.len() as u64);
    for value in &state.answer_features {
        hash_u64(&mut hasher, value.to_bits());
    }
    hash_u64(&mut hasher, state.phi_gradient.len() as u64);
    for value in &state.phi_gradient {
        hash_u64(&mut hasher, value.to_bits());
    }
    hash_u64(&mut hasher, state.chain_length as u64);
    hash_u64(&mut hasher, state.total_phi.to_bits());
    hasher.finalize().to_hex().to_string()
}

fn execution_commitment(
    step: usize,
    source: QualificationActionSource,
    policy_action: &ReasoningAction,
    executed_action: &ReasoningAction,
    encoding_digest: &str,
    pre_state_digest: &str,
    post_state_digest: &str,
    local_transition_contribution: f64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, EXECUTION_DOMAIN);
    hash_u64(&mut hasher, step as u64);
    hash_u64(
        &mut hasher,
        match source {
            QualificationActionSource::Policy => 0,
            QualificationActionSource::Forced => 1,
        },
    );
    hash_str(&mut hasher, &policy_action.primitive_name);
    hash_u64(&mut hasher, transformation_tag(policy_action.transformation));
    hash_str(&mut hasher, &executed_action.primitive_name);
    hash_u64(
        &mut hasher,
        transformation_tag(executed_action.transformation),
    );
    hash_str(&mut hasher, encoding_digest);
    hash_str(&mut hasher, pre_state_digest);
    hash_str(&mut hasher, post_state_digest);
    hash_u64(&mut hasher, local_transition_contribution.to_bits());
    hasher.finalize().to_hex().to_string()
}

fn receipt_commitment(receipt: &AdaptiveQualificationReceipt) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, RECEIPT_DOMAIN);
    hash_str(&mut hasher, &receipt.control_version);
    hash_u64(&mut hasher, receipt.rng_seed);
    hash_u64(&mut hasher, receipt.max_steps as u64);
    hash_str(&mut hasher, &receipt.question_digest);
    hash_str(&mut hasher, &receipt.learner_state_before);
    hash_str(&mut hasher, &receipt.learner_state_after);
    hash_str(&mut hasher, &receipt.final_state_digest);
    hash_u64(&mut hasher, receipt.forced_actions_requested as u64);
    hash_u64(&mut hasher, receipt.forced_actions_applied as u64);
    hash_u64(&mut hasher, receipt.actions.len() as u64);
    for action in &receipt.actions {
        hash_str(&mut hasher, &action.execution_commitment);
    }
    hasher.finalize().to_hex().to_string()
}

fn binary_hv_digest(hv: &BinaryHV) -> String {
    blake3::hash(&hv.0).to_hex().to_string()
}

const fn transformation_tag(transformation: TransformationType) -> u64 {
    match transformation {
        TransformationType::Bind => 0,
        TransformationType::Bundle => 1,
        TransformationType::Permute => 2,
        TransformationType::Resonate => 3,
        TransformationType::Abstract => 4,
        TransformationType::Ground => 5,
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn hash_bool(hasher: &mut blake3::Hasher, value: bool) {
    hasher.update(&[u8::from(value)]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::primitive_system::PrimitiveTier;

    fn reasoner() -> AdaptiveReasoner {
        AdaptiveReasoner::new(PrimitiveTier::NSM)
    }

    #[test]
    fn frozen_run_preserves_learner_state_and_exact_budget() {
        let reasoner = reasoner();
        let before = reasoner.adaptive_qualification_learner_state_commitment();
        let config = AdaptiveQualificationConfig::frozen(5, 0xA11CE);
        let run = reasoner
            .reason_adaptive_qualification_frozen(BinaryHV::random(7001), &config)
            .unwrap();
        let after = reasoner.adaptive_qualification_learner_state_commitment();
        assert_eq!(before, after);
        assert!(run.receipt.learner_state_unchanged());
        assert_eq!(run.chain.executions.len(), 5);
        assert_eq!(run.receipt.actions.len(), 5);
        assert_eq!(run.receipt.forced_actions_applied, 0);
    }

    #[test]
    fn same_seed_input_and_state_reproduce_actions_and_final_state() {
        let reasoner = reasoner();
        let question = BinaryHV::random(7002);
        let config = AdaptiveQualificationConfig::frozen(6, 42);
        let a = reasoner
            .reason_adaptive_qualification_frozen(question, &config)
            .unwrap();
        let b = reasoner
            .reason_adaptive_qualification_frozen(question, &config)
            .unwrap();
        assert_eq!(a.receipt.actions, b.receipt.actions);
        assert_eq!(a.receipt.final_state_digest, b.receipt.final_state_digest);
        assert_eq!(a.receipt.receipt_commitment, b.receipt.receipt_commitment);
    }

    #[test]
    fn forced_action_overrides_execution_but_preserves_policy_observation() {
        let reasoner = reasoner();
        let mut config = AdaptiveQualificationConfig::frozen(4, 77);
        config.forced_actions.push(ForcedAdaptiveAction {
            step: 1,
            primitive_name: "ground".into(),
            transformation: TransformationType::Bundle,
        });
        let run = reasoner
            .reason_adaptive_qualification_frozen(BinaryHV::random(7003), &config)
            .unwrap();
        let record = &run.receipt.actions[1];
        assert_eq!(record.source, QualificationActionSource::Forced);
        assert_eq!(record.executed_action.primitive_name, "ground");
        assert_eq!(record.executed_action.transformation, TransformationType::Bundle);
        assert!(!record.policy_action.primitive_name.is_empty());
        assert_eq!(run.receipt.forced_actions_requested, 1);
        assert_eq!(run.receipt.forced_actions_applied, 1);
    }

    #[test]
    fn duplicate_forced_step_fails_closed() {
        let reasoner = reasoner();
        let config = AdaptiveQualificationConfig {
            max_steps: 3,
            rng_seed: 1,
            forced_actions: vec![
                ForcedAdaptiveAction {
                    step: 1,
                    primitive_name: "ground".into(),
                    transformation: TransformationType::Bundle,
                },
                ForcedAdaptiveAction {
                    step: 1,
                    primitive_name: "compose".into(),
                    transformation: TransformationType::Abstract,
                },
            ],
        };
        assert!(matches!(
            reasoner.reason_adaptive_qualification_frozen(BinaryHV::random(7004), &config),
            Err(AdaptiveQualificationError::DuplicateForcedStep { step: 1 })
        ));
    }

    #[test]
    fn unknown_forced_primitive_fails_closed() {
        let reasoner = reasoner();
        let config = AdaptiveQualificationConfig {
            max_steps: 2,
            rng_seed: 2,
            forced_actions: vec![ForcedAdaptiveAction {
                step: 0,
                primitive_name: "not-a-real-adaptive-primitive".into(),
                transformation: TransformationType::Bind,
            }],
        };
        assert!(matches!(
            reasoner.reason_adaptive_qualification_frozen(BinaryHV::random(7005), &config),
            Err(AdaptiveQualificationError::UnknownForcedPrimitive { .. })
        ));
    }

    #[test]
    fn transformations_outside_live_adaptive_action_space_fail_closed() {
        let reasoner = reasoner();
        let config = AdaptiveQualificationConfig {
            max_steps: 2,
            rng_seed: 3,
            forced_actions: vec![ForcedAdaptiveAction {
                step: 0,
                primitive_name: "ground".into(),
                transformation: TransformationType::Ground,
            }],
        };
        assert!(matches!(
            reasoner.reason_adaptive_qualification_frozen(BinaryHV::random(7006), &config),
            Err(AdaptiveQualificationError::UnsupportedForcedTransformation { .. })
        ));
    }

    #[test]
    fn receipt_is_sensitive_to_forced_intervention() {
        let reasoner = reasoner();
        let question = BinaryHV::random(7007);
        let control = AdaptiveQualificationConfig::frozen(4, 99);
        let mut intervention = control.clone();
        intervention.forced_actions.push(ForcedAdaptiveAction {
            step: 0,
            primitive_name: "compose".into(),
            transformation: TransformationType::Bundle,
        });
        let a = reasoner
            .reason_adaptive_qualification_frozen(question, &control)
            .unwrap();
        let b = reasoner
            .reason_adaptive_qualification_frozen(question, &intervention)
            .unwrap();
        assert_ne!(a.receipt.receipt_commitment, b.receipt.receipt_commitment);
        assert_eq!(a.receipt.learner_state_before, b.receipt.learner_state_before);
    }
}
