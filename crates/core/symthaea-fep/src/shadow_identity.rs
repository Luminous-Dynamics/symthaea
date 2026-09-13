// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test-only qualification of FEP shadow RNG/timestamp non-interference.
//!
//! This module is compiled only under `cfg(test)`. It deliberately exposes no
//! runtime or downstream API for observing the private xorshift continuation.
//! The raw continuation exists only inside qualification so evidence gathering
//! cannot become a prediction side channel in production.

use crate::agent::ActiveInferenceAgent;
use crate::free_energy::ExpectedFreeEnergyComputer;
use crate::generative_model::GenerativeModel;
use crate::shadow_policy::{
    FepShadowPolicyEvaluationV1, FepShadowPolicyReceiptV1, FepShadowPolicyUnavailableV1,
    shadow_policy_evaluation_v1,
};
use crate::types::HiddenState;

const RNG_CONTINUATION_PROBE_DRAWS_V1: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
struct RngContinuationWitnessV1 {
    action_domain_size: usize,
    draws: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum ShadowIdentityUnavailableV1 {
    Base(FepShadowPolicyUnavailableV1),
    InvalidProbeDrawCount { draws: usize },
    InsufficientActionDomain { actions: usize },
    NonUniformProbePolicy { max_delta: f64 },
}

#[derive(Debug, Clone, PartialEq)]
struct ShadowIdentityReceiptV1 {
    base: FepShadowPolicyReceiptV1,
    fep_timestamp: u64,
    perception_cycles: u64,
    rng_continuation: RngContinuationWitnessV1,
}

fn rng_continuation_witness_v1(
    agent: &ActiveInferenceAgent,
    draws: usize,
) -> Result<RngContinuationWitnessV1, ShadowIdentityUnavailableV1> {
    if draws == 0 || draws > 256 {
        return Err(ShadowIdentityUnavailableV1::InvalidProbeDrawCount { draws });
    }
    if agent.config.num_actions < 2 {
        return Err(ShadowIdentityUnavailableV1::InsufficientActionDomain {
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
            return Err(ShadowIdentityUnavailableV1::NonUniformProbePolicy { max_delta });
        }
        continuation.push(result.action);
    }

    Ok(RngContinuationWitnessV1 {
        action_domain_size: agent.config.num_actions,
        draws: continuation,
    })
}

fn shadow_identity_receipt_v1(
    agent: &ActiveInferenceAgent,
) -> Result<ShadowIdentityReceiptV1, ShadowIdentityUnavailableV1> {
    let base = match shadow_policy_evaluation_v1(agent) {
        FepShadowPolicyEvaluationV1::Available(receipt) => receipt,
        FepShadowPolicyEvaluationV1::Unavailable(reason) => {
            return Err(ShadowIdentityUnavailableV1::Base(reason));
        }
    };

    let rng_continuation = rng_continuation_witness_v1(agent, RNG_CONTINUATION_PROBE_DRAWS_V1)?;

    // `act()` returns the exact private FEP timestamp without incrementing it.
    // Run it only on a discarded clone so its compatibility-commitment side
    // effects cannot touch the supplied live agent.
    let mut timestamp_probe = agent.clone();
    let fep_timestamp = timestamp_probe.act(0).timestamp;

    Ok(ShadowIdentityReceiptV1 {
        base,
        fep_timestamp,
        perception_cycles: agent.stats.perception_cycles,
        rng_continuation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ActiveInferenceAgentConfig;
    use crate::types::Observation;

    fn agent_with_seed(seed: u64) -> ActiveInferenceAgent {
        let mut agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig::default());
        agent.set_rng_seed(seed);
        agent
    }

    #[test]
    fn repeated_shadow_queries_preserve_timestamp_and_rng_continuation() {
        let agent = agent_with_seed(0xA11CE5EED);

        let before = shadow_identity_receipt_v1(&agent).expect("first identity receipt");
        let again = shadow_identity_receipt_v1(&agent).expect("repeated identity receipt");

        assert_eq!(before, again);
    }

    #[test]
    fn one_live_selection_shifts_rng_continuation_by_exactly_one_draw() {
        let mut agent = agent_with_seed(0x5157_1E57);

        let before = rng_continuation_witness_v1(&agent, RNG_CONTINUATION_PROBE_DRAWS_V1 + 1)
            .expect("pre-selection continuation witness");

        let _ = agent.select_action();

        let after = rng_continuation_witness_v1(&agent, RNG_CONTINUATION_PROBE_DRAWS_V1)
            .expect("post-selection continuation witness");

        assert_eq!(&before.draws[1..], after.draws.as_slice());
    }

    #[test]
    fn v1_shadow_query_does_not_shift_rng_continuation() {
        let agent = agent_with_seed(0x5A0D_0BAD);

        let before = rng_continuation_witness_v1(&agent, RNG_CONTINUATION_PROBE_DRAWS_V1)
            .expect("pre-shadow continuation witness");
        let _ = shadow_policy_evaluation_v1(&agent);
        let after = rng_continuation_witness_v1(&agent, RNG_CONTINUATION_PROBE_DRAWS_V1)
            .expect("post-shadow continuation witness");

        assert_eq!(before, after);
    }

    #[test]
    fn perception_advances_timestamp_while_shadow_queries_do_not() {
        let mut agent = agent_with_seed(0x71A1_E57A);

        let before = shadow_identity_receipt_v1(&agent).expect("pre-perception receipt");
        let repeated = shadow_identity_receipt_v1(&agent).expect("repeated receipt");
        assert_eq!(before, repeated);

        let observation = Observation::new(
            vec![0.25; agent.config.obs_dim],
            1.0,
            "shadow-identity-timestamp",
        );
        let _ = agent.perceive(&observation);

        let after = shadow_identity_receipt_v1(&agent).expect("post-perception receipt");
        assert_eq!(after.fep_timestamp, before.fep_timestamp + 1);
        assert_eq!(after.perception_cycles, before.perception_cycles + 1);
    }

    #[test]
    fn identity_receipt_wraps_the_exact_v1_shadow_receipt() {
        let agent = agent_with_seed(0xBACE_F00D);

        let v1 = match shadow_policy_evaluation_v1(&agent) {
            FepShadowPolicyEvaluationV1::Available(receipt) => receipt,
            FepShadowPolicyEvaluationV1::Unavailable(reason) => {
                panic!("shadow V1 unexpectedly unavailable: {reason:?}")
            }
        };
        let identity = shadow_identity_receipt_v1(&agent).expect("identity receipt");

        assert_eq!(identity.base, v1);
    }

    #[test]
    fn shadow_on_one_clone_preserves_live_selection_and_future_rng_identity() {
        let base = agent_with_seed(0xC10E_C10E);
        let mut control = base.clone();
        let mut shadow = base;

        let _ = shadow_policy_evaluation_v1(&shadow);

        let control_selection = control.select_action();
        let shadow_selection = shadow.select_action();

        assert_eq!(control_selection.action, shadow_selection.action);
        assert_eq!(
            control_selection.action_probabilities,
            shadow_selection.action_probabilities
        );

        let control_future =
            rng_continuation_witness_v1(&control, RNG_CONTINUATION_PROBE_DRAWS_V1)
                .expect("control continuation witness");
        let shadow_future =
            rng_continuation_witness_v1(&shadow, RNG_CONTINUATION_PROBE_DRAWS_V1)
                .expect("shadow continuation witness");
        assert_eq!(control_future, shadow_future);
    }

    #[test]
    fn continuation_witness_is_independent_of_live_policy_state() {
        let a = agent_with_seed(0xDEC0_DED0);
        let mut b = a.clone();

        for _ in 0..17 {
            b.efe_computer.record_committed_action(0);
        }
        if let Some(first) = b.belief.mean.first_mut() {
            *first = 0.99;
        }

        let wa = rng_continuation_witness_v1(&a, RNG_CONTINUATION_PROBE_DRAWS_V1)
            .expect("baseline continuation witness");
        let wb = rng_continuation_witness_v1(&b, RNG_CONTINUATION_PROBE_DRAWS_V1)
            .expect("perturbed-policy continuation witness");

        assert_eq!(wa, wb);
    }

    #[test]
    fn continuation_witness_rejects_single_action_domain() {
        let mut config = ActiveInferenceAgentConfig::default();
        config.num_actions = 1;
        let agent = ActiveInferenceAgent::new(config);

        assert_eq!(
            rng_continuation_witness_v1(&agent, RNG_CONTINUATION_PROBE_DRAWS_V1),
            Err(ShadowIdentityUnavailableV1::InsufficientActionDomain { actions: 1 })
        );
    }
}
