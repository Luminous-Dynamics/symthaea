use symthaea_fep::shadow_policy::{
    FepShadowPolicyEvaluationV1, FepShadowPolicyUnavailableV1, shadow_policy_evaluation_v1,
    transition_evidence_source_state_v1,
};
use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, GenerativeModel, HiddenState, Observation,
    TemporalDifferenceLearner, TemporalDifferenceLearningConfig,
};

fn test_config() -> ActiveInferenceAgentConfig {
    ActiveInferenceAgentConfig {
        state_dim: 2,
        obs_dim: 2,
        num_actions: 2,
        inference_iterations: 1,
        belief_learning_rate: 0.05,
        planning_horizon: 1,
        action_temperature: 0.7,
        enable_model_learning: true,
        enable_td_learning: true,
        td_config: TemporalDifferenceLearningConfig::default(),
    }
}

fn make_agent() -> ActiveInferenceAgent {
    let mut agent = ActiveInferenceAgent::new(test_config());
    agent.belief.mean = vec![0.7, 0.3];
    agent.belief.precision = vec![2.0, 3.0];
    agent.efe_computer.set_preferences(vec![0.6, 0.4], 2.0);
    agent
}

fn available(
    evaluation: FepShadowPolicyEvaluationV1,
) -> symthaea_fep::shadow_policy::FepShadowPolicyReceiptV1 {
    match evaluation {
        FepShadowPolicyEvaluationV1::Available(receipt) => receipt,
        FepShadowPolicyEvaluationV1::Unavailable(reason) => {
            panic!("shadow unexpectedly unavailable: {reason:?}")
        }
    }
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "left={left:.16e}, right={right:.16e}, tolerance={tolerance:.3e}"
    );
}

fn public_state_json(agent: &ActiveInferenceAgent) -> String {
    serde_json::to_string(&(
        &agent.config,
        &agent.belief,
        &agent.previous_state,
        &agent.last_action,
        &agent.model,
        &agent.free_energy_calc,
        &agent.precision,
        &agent.efe_computer,
        &agent.td_learner,
        &agent.last_fe_components,
        &agent.stats,
    ))
    .expect("public agent state should serialize")
}

#[test]
fn transition_source_adapter_preserves_finite_last_max_tie_semantics() {
    let state = HiddenState {
        mean: vec![0.8, 0.8, 0.3, 0.8],
        precision: vec![1.0; 4],
        mode_probs: vec![1.0],
        current_mode: 0,
    };

    let source = transition_evidence_source_state_v1(&state, 4)
        .expect("finite tied state should have a source row");
    assert_eq!(source.index, 3, "last maximal component must win ties");
}

#[test]
fn transition_source_adapter_fails_closed_on_malformed_state() {
    let empty = HiddenState {
        mean: vec![],
        precision: vec![],
        mode_probs: vec![1.0],
        current_mode: 0,
    };
    assert!(matches!(
        transition_evidence_source_state_v1(&empty, 0),
        Err(FepShadowPolicyUnavailableV1::EmptyStateDomain)
    ));

    let wrong_dim = HiddenState {
        mean: vec![0.5],
        precision: vec![1.0],
        mode_probs: vec![1.0],
        current_mode: 0,
    };
    assert!(matches!(
        transition_evidence_source_state_v1(&wrong_dim, 2),
        Err(FepShadowPolicyUnavailableV1::BeliefDimensionMismatch { .. })
    ));

    let non_finite = HiddenState {
        mean: vec![0.2, f64::NAN],
        precision: vec![1.0, 1.0],
        mode_probs: vec![1.0],
        current_mode: 0,
    };
    assert!(matches!(
        transition_evidence_source_state_v1(&non_finite, 2),
        Err(FepShadowPolicyUnavailableV1::NonFiniteBeliefMean { index: 1, .. })
    ));
}

#[test]
fn transition_source_adapter_matches_real_td_evidence_row_for_finite_ties() {
    let config = TemporalDifferenceLearningConfig::default();
    let mut learner = TemporalDifferenceLearner::new(config, 2, 3, 2);
    let mut model = GenerativeModel::new(3, 2, 2);

    let old_state = HiddenState {
        mean: vec![0.6, 0.6, 0.2],
        precision: vec![1.0; 3],
        mode_probs: vec![1.0],
        current_mode: 0,
    };
    let new_state = HiddenState {
        mean: vec![0.1, 0.2, 0.9],
        precision: vec![1.0; 3],
        mode_probs: vec![1.0],
        current_mode: 0,
    };
    let observation = Observation::new(vec![0.2, 0.8], 1.0, "shadow-source-test");

    let source = transition_evidence_source_state_v1(&old_state, 3)
        .expect("finite state should map to evidence row");
    assert_eq!(source.index, 1, "finite tie should use later maximal index");

    learner.update_model(
        &mut model,
        &old_state,
        0,
        &new_state,
        &observation,
        0.0,
    );

    assert_eq!(
        learner.confidence_tracker.transition_counts[0][source.index][2],
        1,
        "versioned source adapter must identify the exact row production TD evidence mutates"
    );
    assert_eq!(learner.confidence_tracker.transition_counts[0][0][2], 0);
}

#[test]
fn shadow_legacy_probabilities_match_live_agent_pre_sampling_vector() {
    let agent = make_agent();
    let receipt = available(shadow_policy_evaluation_v1(&agent));

    let mut live = agent.clone();
    let selection = live.select_action();

    assert_eq!(receipt.legacy_actions.len(), selection.action_probabilities.len());
    for action in 0..selection.action_probabilities.len() {
        assert_eq!(receipt.legacy_actions[action].action, action);
        assert_close(
            receipt.legacy_actions[action].probability,
            selection.action_probabilities[action],
            1e-15,
        );
    }
}

#[test]
fn missing_transition_evidence_is_typed_unavailable_not_fabricated_counts() {
    let mut config = test_config();
    config.enable_td_learning = false;
    let agent = ActiveInferenceAgent::new(config);

    assert!(matches!(
        shadow_policy_evaluation_v1(&agent),
        FepShadowPolicyEvaluationV1::Unavailable(
            FepShadowPolicyUnavailableV1::MissingTransitionEvidence
        )
    ));
}

#[test]
fn repeated_shadow_queries_are_referentially_pure() {
    let agent = make_agent();
    let before = public_state_json(&agent);

    let first = shadow_policy_evaluation_v1(&agent);
    let second = shadow_policy_evaluation_v1(&agent);

    assert_eq!(first, second);
    assert_eq!(public_state_json(&agent), before);
}

#[test]
fn shadow_queries_do_not_change_real_selection_or_learning_trajectory() {
    let mut control = make_agent();
    let mut shadow = control.clone();
    control.set_rng_seed(0x5EED_1234_5678_9ABC);
    shadow.set_rng_seed(0x5EED_1234_5678_9ABC);

    for step in 0..24usize {
        let shadow_before = public_state_json(&shadow);
        let receipt = available(shadow_policy_evaluation_v1(&shadow));
        assert_eq!(public_state_json(&shadow), shadow_before);

        let control_selection = control.select_action();
        let shadow_selection = shadow.select_action();
        assert_eq!(control_selection.action, shadow_selection.action);
        assert_eq!(
            control_selection.action_probabilities.len(),
            shadow_selection.action_probabilities.len()
        );
        for (control_p, shadow_p) in control_selection
            .action_probabilities
            .iter()
            .zip(shadow_selection.action_probabilities.iter())
        {
            assert_close(*control_p, *shadow_p, 0.0);
        }

        for (legacy, live_probability) in receipt
            .legacy_actions
            .iter()
            .zip(control_selection.action_probabilities.iter())
        {
            assert_close(legacy.probability, *live_probability, 1e-15);
        }

        let control_outcome = control.act(control_selection.action);
        let shadow_outcome = shadow.act(shadow_selection.action);
        assert_eq!(control_outcome.action, shadow_outcome.action);
        assert_eq!(
            control_outcome.expected_observation,
            shadow_outcome.expected_observation
        );

        let phase = step as f64 / 23.0;
        let observation = Observation::new(
            vec![0.25 + 0.5 * phase, 0.75 - 0.5 * phase],
            1.0,
            "shadow-zero-interference-sequence",
        );
        let control_perception = control.perceive(&observation);
        let shadow_perception = shadow.perceive(&observation);
        assert_close(control_perception.belief_change, shadow_perception.belief_change, 0.0);
        assert_eq!(
            control_perception.updated_belief.mean,
            shadow_perception.updated_belief.mean
        );
    }

    assert_eq!(
        public_state_json(&control),
        public_state_json(&shadow),
        "shadow telemetry must not alter any serialized public agent state after the same trajectory"
    );
}
