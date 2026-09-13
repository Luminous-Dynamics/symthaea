// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only evidence identity for the FEP shadow policy.
//!
//! This module strengthens shadow-policy provenance without exposing mutation
//! authority over the live agent. V2 wraps the frozen V1 shadow receipt with:
//!
//! - an exact FEP timestamp witness read through a discarded clone; and
//! - a deterministic RNG-continuation witness generated from a discarded clone
//!   under an action-neutral probe policy.
//!
//! The continuation witness is intentionally **not** described as the raw private
//! RNG word. It is a black-box identity of the future xorshift continuation under
//! a fixed neutral policy. Shadow evaluation itself remains read-only with respect
//! to the supplied live agent.

use crate::agent::ActiveInferenceAgent;
use crate::free_energy::ExpectedFreeEnergyComputer;
use crate::generative_model::GenerativeModel;
use crate::shadow_policy::{
    FepShadowPolicyEvaluationV1, FepShadowPolicyReceiptV1, FepShadowPolicyUnavailableV1,
    shadow_policy_evaluation_v1,
};
use crate::types::HiddenState;

pub const FEP_SHADOW_POLICY_SCHEMA_V2: u16 = 2;
pub const RNG_CONTINUATION_WITNESS_SCHEMA_V1: u16 = 1;
pub const RNG_CONTINUATION_PROBE_DRAWS_V1: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RngContinuationWitnessV1 {
    pub schema_version: u16,
    pub action_domain_size: usize,
    pub draws: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FepShadowPolicyReceiptV2 {
    pub schema_version: u16,
    pub base: FepShadowPolicyReceiptV1,
    pub fep_timestamp: u64,
    pub perception_cycles: u64,
    pub rng_continuation: RngContinuationWitnessV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FepShadowPolicyUnavailableV2 {
    Base(FepShadowPolicyUnavailableV1),
    InvalidProbeDrawCount { draws: usize },
    InsufficientActionDomain { actions: usize },
    NonUniformProbePolicy { max_delta: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FepShadowPolicyEvaluationV2 {
    Available(FepShadowPolicyReceiptV2),
    Unavailable(FepShadowPolicyUnavailableV2),
}

pub fn rng_continuation_witness_v1(
    agent: &ActiveInferenceAgent,
    draws: usize,
) -> Result<RngContinuationWitnessV1, FepShadowPolicyUnavailableV2> {
    if draws == 0 || draws > 256 {
        return Err(FepShadowPolicyUnavailableV2::InvalidProbeDrawCount { draws });
    }
    if agent.config.num_actions < 2 {
        return Err(FepShadowPolicyUnavailableV2::InsufficientActionDomain {
            actions: agent.config.num_actions,
        });
    }

    let mut probe = agent.clone();
    probe.model = GenerativeModel::new(
        agent.config.state_dim,
        agent.config.obs_dim,
        agent.config.num_actions,
    );
    probe.belief = HiddenState::new(agent.config.state_dim);
    probe.previous_state = None;
    probe.last_action = None;
    probe.efe_computer = ExpectedFreeEnergyComputer::new(agent.config.obs_dim);
    probe.config.action_temperature = 1.0;
    probe.stats = Default::default();

    let expected_probability = 1.0 / agent.config.num_actions as f64;
    let mut continuation = Vec::with_capacity(draws);

    for _ in 0..draws {
        let result = probe.select_action();
        let max_delta = result
            .action_probabilities
            .iter()
            .map(|probability| (probability - expected_probability).abs())
            .fold(0.0_f64, f64::max);
        if max_delta > 1e-12 {
            return Err(FepShadowPolicyUnavailableV2::NonUniformProbePolicy { max_delta });
        }
        continuation.push(result.action);
    }

    Ok(RngContinuationWitnessV1 {
        schema_version: RNG_CONTINUATION_WITNESS_SCHEMA_V1,
        action_domain_size: agent.config.num_actions,
        draws: continuation,
    })
}

pub fn shadow_policy_evaluation_v2(agent: &ActiveInferenceAgent) -> FepShadowPolicyEvaluationV2 {
    let base = match shadow_policy_evaluation_v1(agent) {
        FepShadowPolicyEvaluationV1::Available(receipt) => receipt,
        FepShadowPolicyEvaluationV1::Unavailable(reason) => {
            return FepShadowPolicyEvaluationV2::Unavailable(
                FepShadowPolicyUnavailableV2::Base(reason),
            );
        }
    };

    let rng_continuation = match rng_continuation_witness_v1(
        agent,
        RNG_CONTINUATION_PROBE_DRAWS_V1,
    ) {
        Ok(value) => value,
        Err(reason) => return FepShadowPolicyEvaluationV2::Unavailable(reason),
    };

    let mut timestamp_probe = agent.clone();
    let fep_timestamp = timestamp_probe.act(0).timestamp;

    FepShadowPolicyEvaluationV2::Available(FepShadowPolicyReceiptV2 {
        schema_version: FEP_SHADOW_POLICY_SCHEMA_V2,
        base,
        fep_timestamp,
        perception_cycles: agent.stats.perception_cycles,
        rng_continuation,
    })
}
