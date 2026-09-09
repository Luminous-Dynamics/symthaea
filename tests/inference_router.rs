#[path = "../src/language/inference_contract.rs"]
mod inference_contract;
#[path = "../src/language/inference_permit.rs"]
mod inference_permit;
#[path = "../src/language/inference_binding.rs"]
mod inference_binding;
#[path = "../src/language/inference_receipt.rs"]
mod inference_receipt;
#[path = "../src/language/inference_resource_guard.rs"]
mod inference_resource_guard;
#[path = "../src/language/inference_router.rs"]
mod inference_router;

use inference_contract::{
    ExecutionLocation, InferenceCandidate, InferencePolicy, InferencePurpose, InferenceRequest,
    InferenceRequirements, InformationClass, ModelIdentity, ProviderRetentionPolicy,
    ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_resource_guard::{
    InferenceResourceAuthority, InferenceResourceGuard, InferenceResourceGuardConfig,
    InferenceResourceGuardError,
};
use inference_router::{
    InferenceRouteMetrics, InferenceRouteRejectionReason, InferenceRouterError,
    InferenceRouterWeights, InferenceRoutingCandidate, plan_inference_routes,
};

fn request(class: InformationClass) -> InferenceRequest {
    let mut requirements =
        InferenceRequirements::minimal(InferencePurpose::GeneralReasoning, class);
    requirements.estimated_input_tokens = 100;
    requirements.max_output_tokens = 200;
    InferenceRequest::new("route this safely", requirements)
}

fn local_candidate(provider: &str, quality_id: &str) -> InferenceCandidate {
    InferenceCandidate {
        provider_id: provider.to_string(),
        model: ModelIdentity::ContentVerified {
            digest: quality_id.to_string(),
        },
        location: ExecutionLocation::LocalDevice,
        supported_purposes: vec![InferencePurpose::GeneralReasoning],
        context_window_tokens: 8_192,
        supports_streaming: true,
        supports_tools: false,
        supports_structured_output: false,
        training_policy: ProviderTrainingPolicy::Never,
        retention_policy: ProviderRetentionPolicy::ZeroRetention,
        routing_policy: ProviderRoutingPolicy::DirectOnly,
        max_charge_microusd: Some(0),
    }
}

fn remote_candidate(provider: &str, model: &str, charge: u64) -> InferenceCandidate {
    InferenceCandidate {
        provider_id: provider.to_string(),
        model: ModelIdentity::ProviderAttested {
            provider: provider.to_string(),
            declared_model: model.to_string(),
        },
        location: ExecutionLocation::RemoteProvider,
        supported_purposes: vec![InferencePurpose::GeneralReasoning],
        context_window_tokens: 8_192,
        supports_streaming: true,
        supports_tools: false,
        supports_structured_output: false,
        training_policy: ProviderTrainingPolicy::Never,
        retention_policy: ProviderRetentionPolicy::ZeroRetention,
        routing_policy: ProviderRoutingPolicy::DirectOnly,
        max_charge_microusd: Some(charge),
    }
}

fn guard(instance: u8, requests: u64, tokens: u64) -> InferenceResourceGuard {
    InferenceResourceGuard::new(
        InferenceResourceAuthority::new(1, 1, Some(requests), Some(tokens)),
        InferenceResourceGuardConfig::default(),
        [instance; 32],
    )
    .unwrap()
}

fn metrics(quality: u16, reliability: u16, latency: u64) -> InferenceRouteMetrics {
    InferenceRouteMetrics::new(Some(quality), Some(reliability), Some(latency)).unwrap()
}

#[test]
fn input_order_does_not_change_ranked_plan() {
    let a = local_candidate("a-provider", "a-model");
    let b = local_candidate("b-provider", "b-model");
    let req = request(InformationClass::Public);
    let policy = InferencePolicy::sovereign_default();
    let m = metrics(700, 700, 10);

    let first = plan_inference_routes(
        &policy,
        &req,
        &[
            InferenceRoutingCandidate { candidate: &b, resource_guard: None, metrics: m },
            InferenceRoutingCandidate { candidate: &a, resource_guard: None, metrics: m },
        ],
        InferenceRouterWeights::default(),
        0,
    )
    .unwrap();
    let second = plan_inference_routes(
        &policy,
        &req,
        &[
            InferenceRoutingCandidate { candidate: &a, resource_guard: None, metrics: m },
            InferenceRoutingCandidate { candidate: &b, resource_guard: None, metrics: m },
        ],
        InferenceRouterWeights::default(),
        0,
    )
    .unwrap();

    let first_keys: Vec<_> = first.ranked().iter().map(|route| route.stable_key()).collect();
    let second_keys: Vec<_> = second.ranked().iter().map(|route| route.stable_key()).collect();
    assert_eq!(first_keys, second_keys);
    assert!(first_keys[0].starts_with("a-provider"));
}

#[test]
fn sovereign_policy_cannot_be_outvoted_by_remote_quality() {
    let local = local_candidate("local", "local-digest");
    let remote = remote_candidate("remote", "huge-model", 0);
    let remote_guard = guard(2, 1_000, 1_000_000);
    let req = request(InformationClass::Public);

    let plan = plan_inference_routes(
        &InferencePolicy::sovereign_default(),
        &req,
        &[
            InferenceRoutingCandidate {
                candidate: &remote,
                resource_guard: Some(&remote_guard),
                metrics: metrics(1_000, 1_000, 1),
            },
            InferenceRoutingCandidate {
                candidate: &local,
                resource_guard: None,
                metrics: metrics(1, 1, 10_000),
            },
        ],
        InferenceRouterWeights {
            quality: u16::MAX,
            reliability: u16::MAX,
            locality: 0,
            resource_headroom: 0,
            cost_efficiency: 0,
            latency: 0,
        },
        0,
    )
    .unwrap();

    assert_eq!(plan.ranked().len(), 1);
    assert_eq!(plan.preferred().unwrap().candidate().provider_id, "local");
    assert_eq!(plan.rejected().len(), 1);
    assert!(matches!(
        plan.rejected()[0].reason,
        InferenceRouteRejectionReason::Admission(_)
    ));
}

#[test]
fn raw_cognitive_state_can_only_survive_on_local_route() {
    let local = local_candidate("local", "local-digest");
    let remote = remote_candidate("remote", "reasoner", 0);
    let remote_guard = guard(3, 100, 100_000);
    let req = request(InformationClass::CognitiveState);

    let plan = plan_inference_routes(
        &InferencePolicy {
            allow_remote: true,
            allow_provider_training: true,
            allow_provider_retention: true,
            allow_third_party_routing: true,
            max_charge_microusd: u64::MAX,
        },
        &req,
        &[
            InferenceRoutingCandidate { candidate: &remote, resource_guard: Some(&remote_guard), metrics: metrics(1_000, 1_000, 1) },
            InferenceRoutingCandidate { candidate: &local, resource_guard: None, metrics: metrics(100, 100, 100) },
        ],
        InferenceRouterWeights::default(),
        0,
    )
    .unwrap();

    assert_eq!(plan.ranked().len(), 1);
    assert_eq!(plan.preferred().unwrap().candidate().provider_id, "local");
    assert_eq!(plan.rejected()[0].provider_id, "remote");
}

#[test]
fn remote_route_requires_resource_state() {
    let remote = remote_candidate("remote", "reasoner", 0);
    let req = request(InformationClass::RemoteSafeDerived);
    let plan = plan_inference_routes(
        &InferencePolicy::free_private(),
        &req,
        &[InferenceRoutingCandidate {
            candidate: &remote,
            resource_guard: None,
            metrics: metrics(1_000, 1_000, 1),
        }],
        InferenceRouterWeights::default(),
        0,
    )
    .unwrap();
    assert!(plan.ranked().is_empty());
    assert_eq!(
        plan.rejected()[0].reason,
        InferenceRouteRejectionReason::ResourceStateRequired
    );
}

#[test]
fn scarce_quota_is_conserved_as_a_soft_preference() {
    let scarce = remote_candidate("scarce", "same-quality", 0);
    let roomy = remote_candidate("roomy", "same-quality", 0);
    let scarce_guard = guard(4, 1, 10_000);
    let roomy_guard = guard(5, 50, 100_000);
    let req = request(InformationClass::RemoteSafeDerived);
    let m = metrics(800, 800, 50);

    let plan = plan_inference_routes(
        &InferencePolicy::free_private(),
        &req,
        &[
            InferenceRoutingCandidate { candidate: &scarce, resource_guard: Some(&scarce_guard), metrics: m },
            InferenceRoutingCandidate { candidate: &roomy, resource_guard: Some(&roomy_guard), metrics: m },
        ],
        InferenceRouterWeights {
            quality: 0,
            reliability: 0,
            locality: 0,
            resource_headroom: 10,
            cost_efficiency: 0,
            latency: 0,
        },
        0,
    )
    .unwrap();

    assert_eq!(plan.ranked().len(), 2);
    assert_eq!(plan.ranked()[0].candidate().provider_id, "roomy");
    assert!(
        plan.ranked()[0].score().resource_headroom_milli
            > plan.ranked()[1].score().resource_headroom_milli
    );
}

#[test]
fn remote_quality_can_win_only_after_hard_gates_admit_it() {
    let local = local_candidate("local", "local-digest");
    let remote = remote_candidate("remote", "reasoner", 0);
    let remote_guard = guard(6, 100, 100_000);
    let req = request(InformationClass::RemoteSafeDerived);

    let plan = plan_inference_routes(
        &InferencePolicy::free_private(),
        &req,
        &[
            InferenceRoutingCandidate { candidate: &local, resource_guard: None, metrics: metrics(100, 500, 10) },
            InferenceRoutingCandidate { candidate: &remote, resource_guard: Some(&remote_guard), metrics: metrics(1_000, 500, 10) },
        ],
        InferenceRouterWeights {
            quality: 10,
            reliability: 0,
            locality: 1,
            resource_headroom: 0,
            cost_efficiency: 0,
            latency: 0,
        },
        0,
    )
    .unwrap();

    assert_eq!(plan.preferred().unwrap().candidate().provider_id, "remote");
}

#[test]
fn paid_candidate_is_rejected_by_free_private_before_scoring() {
    let paid = remote_candidate("paid", "reasoner", 1);
    let paid_guard = guard(7, 100, 100_000);
    let req = request(InformationClass::RemoteSafeDerived);
    let plan = plan_inference_routes(
        &InferencePolicy::free_private(),
        &req,
        &[InferenceRoutingCandidate { candidate: &paid, resource_guard: Some(&paid_guard), metrics: metrics(1_000, 1_000, 1) }],
        InferenceRouterWeights::default(),
        0,
    )
    .unwrap();
    assert!(plan.ranked().is_empty());
    assert!(matches!(plan.rejected()[0].reason, InferenceRouteRejectionReason::Admission(_)));
}

#[test]
fn routing_plan_is_not_resource_authority() {
    let remote = remote_candidate("remote", "reasoner", 0);
    let mut remote_guard = guard(8, 1, 10_000);
    let req = request(InformationClass::RemoteSafeDerived);
    let plan = plan_inference_routes(
        &InferencePolicy::free_private(),
        &req,
        &[InferenceRoutingCandidate { candidate: &remote, resource_guard: Some(&remote_guard), metrics: metrics(900, 900, 10) }],
        InferenceRouterWeights::default(),
        0,
    )
    .unwrap();
    assert_eq!(plan.ranked().len(), 1);

    // Another request consumes the resource after planning. The stale plan cannot
    // bypass the final authoritative reservation.
    let _other = remote_guard.reserve(plan.reserved_token_exposure(), 1).unwrap();
    assert_eq!(
        remote_guard.reserve(plan.reserved_token_exposure(), 2).unwrap_err(),
        InferenceResourceGuardError::RequestBudgetExhausted
    );
}

#[test]
fn invalid_metric_and_token_exposure_overflow_fail_explicitly() {
    assert_eq!(
        InferenceRouteMetrics::new(Some(1_001), None, None).unwrap_err(),
        InferenceRouterError::MetricOutOfRange
    );

    let local = local_candidate("local", "digest");
    let mut req = request(InformationClass::Public);
    req.requirements.estimated_input_tokens = u64::MAX;
    req.requirements.max_output_tokens = 1;
    let err = plan_inference_routes(
        &InferencePolicy::sovereign_default(),
        &req,
        &[InferenceRoutingCandidate { candidate: &local, resource_guard: None, metrics: InferenceRouteMetrics::default() }],
        InferenceRouterWeights::default(),
        0,
    )
    .unwrap_err();
    assert_eq!(err, InferenceRouterError::TokenExposureOverflow);
}
