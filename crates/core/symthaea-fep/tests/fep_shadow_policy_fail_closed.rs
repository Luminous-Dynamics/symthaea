use symthaea_fep::shadow_policy::{
    FepShadowPolicyEvaluationV1, FepShadowPolicyUnavailableV1, shadow_policy_evaluation_v1,
};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

#[test]
fn shadow_rejects_state_dimension_drift_between_config_and_model() {
    let mut config = ActiveInferenceAgentConfig::default();
    config.state_dim = 2;
    config.obs_dim = 2;
    config.num_actions = 2;
    let mut agent = ActiveInferenceAgent::new(config);
    agent.config.state_dim = 3;

    assert!(matches!(
        shadow_policy_evaluation_v1(&agent),
        FepShadowPolicyEvaluationV1::Unavailable(
            FepShadowPolicyUnavailableV1::ModelStateDimensionMismatch {
                configured: 3,
                model: 2,
            }
        )
    ));
}

#[test]
fn shadow_rejects_observation_dimension_drift_between_config_and_model() {
    let mut config = ActiveInferenceAgentConfig::default();
    config.state_dim = 2;
    config.obs_dim = 2;
    config.num_actions = 2;
    let mut agent = ActiveInferenceAgent::new(config);
    agent.config.obs_dim = 3;

    assert!(matches!(
        shadow_policy_evaluation_v1(&agent),
        FepShadowPolicyEvaluationV1::Unavailable(
            FepShadowPolicyUnavailableV1::ModelObservationDimensionMismatch {
                configured: 3,
                model: 2,
            }
        )
    ));
}

#[test]
fn shadow_rejects_action_domain_drift_between_config_and_model() {
    let mut config = ActiveInferenceAgentConfig::default();
    config.state_dim = 2;
    config.obs_dim = 2;
    config.num_actions = 2;
    let mut agent = ActiveInferenceAgent::new(config);
    agent.config.num_actions = 3;

    assert!(matches!(
        shadow_policy_evaluation_v1(&agent),
        FepShadowPolicyEvaluationV1::Unavailable(
            FepShadowPolicyUnavailableV1::ModelActionCountMismatch {
                configured: 3,
                model: 2,
            }
        )
    ));
}
