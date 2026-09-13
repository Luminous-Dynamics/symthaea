// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fep::shadow_identity::{
    FepShadowPolicyEvaluationV2, FepShadowPolicyUnavailableV2, RNG_CONTINUATION_PROBE_DRAWS_V1,
    rng_continuation_witness_v1, shadow_policy_evaluation_v2,
};
use symthaea_fep::shadow_policy::{FepShadowPolicyEvaluationV1, shadow_policy_evaluation_v1};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

fn agent_with_seed(seed: u64) -> ActiveInferenceAgent {
    let mut agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig::default());
    agent.set_rng_seed(seed);
    agent
}

fn available_v2(agent: &ActiveInferenceAgent) -> symthaea_fep::shadow_identity::FepShadowPolicyReceiptV2 {
    match shadow_policy_evaluation_v2(agent) {
        FepShadowPolicyEvaluationV2::Available(receipt) => receipt,
        FepShadowPolicyEvaluationV2::Unavailable(reason) => {
            panic!("shadow V2 unexpectedly unavailable: {reason:?}")
        }
    }
}

#[test]
fn repeated_shadow_queries_preserve_timestamp_and_rng_continuation() {
    let agent = agent_with_seed(0xA11CE5EED);

    let before = available_v2(&agent);
    let again = available_v2(&agent);

    assert_eq!(before.fep_timestamp, again.fep_timestamp);
    assert_eq!(before.perception_cycles, again.perception_cycles);
    assert_eq!(before.rng_continuation, again.rng_continuation);
    assert_eq!(before.base, again.base);
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
fn shadow_query_does_not_shift_rng_continuation() {
    let agent = agent_with_seed(0x5A0D_0BAD);

    let before = rng_continuation_witness_v1(&agent, RNG_CONTINUATION_PROBE_DRAWS_V1)
        .expect("pre-shadow continuation witness");
    let _ = available_v2(&agent);
    let after = rng_continuation_witness_v1(&agent, RNG_CONTINUATION_PROBE_DRAWS_V1)
        .expect("post-shadow continuation witness");

    assert_eq!(before, after);
}

#[test]
fn perception_advances_timestamp_while_shadow_queries_do_not() {
    let mut agent = agent_with_seed(0x71A1_E57A);

    let before = available_v2(&agent);
    let repeated = available_v2(&agent);
    assert_eq!(before.fep_timestamp, repeated.fep_timestamp);
    assert_eq!(before.perception_cycles, repeated.perception_cycles);

    let observation = Observation::new(
        vec![0.25; agent.config.obs_dim],
        1.0,
        "shadow-identity-timestamp",
    );
    let _ = agent.perceive(&observation);

    let after = available_v2(&agent);
    assert_eq!(after.fep_timestamp, before.fep_timestamp + 1);
    assert_eq!(after.perception_cycles, before.perception_cycles + 1);
}

#[test]
fn v2_wraps_the_exact_v1_shadow_receipt() {
    let agent = agent_with_seed(0xBACE_F00D);

    let v1 = match shadow_policy_evaluation_v1(&agent) {
        FepShadowPolicyEvaluationV1::Available(receipt) => receipt,
        FepShadowPolicyEvaluationV1::Unavailable(reason) => {
            panic!("shadow V1 unexpectedly unavailable: {reason:?}")
        }
    };
    let v2 = available_v2(&agent);

    assert_eq!(v2.base, v1);
}

#[test]
fn shadow_on_one_clone_preserves_live_selection_and_future_rng_identity() {
    let base = agent_with_seed(0xC10E_C10E);
    let mut control = base.clone();
    let mut shadow = base;

    let _ = available_v2(&shadow);

    let control_selection = control.select_action();
    let shadow_selection = shadow.select_action();

    assert_eq!(control_selection.action, shadow_selection.action);
    assert_eq!(
        control_selection.action_probabilities,
        shadow_selection.action_probabilities
    );

    let control_future = rng_continuation_witness_v1(&control, RNG_CONTINUATION_PROBE_DRAWS_V1)
        .expect("control continuation witness");
    let shadow_future = rng_continuation_witness_v1(&shadow, RNG_CONTINUATION_PROBE_DRAWS_V1)
        .expect("shadow continuation witness");
    assert_eq!(control_future, shadow_future);
}

#[test]
fn continuation_witness_is_independent_of_live_commitment_history() {
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
        Err(FepShadowPolicyUnavailableV2::InsufficientActionDomain { actions: 1 })
    );
}
