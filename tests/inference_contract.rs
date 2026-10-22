// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

// IF-0 intentionally compiles the contract independently before runtime export.
#[path = "../src/language/inference_contract.rs"]
mod inference_contract;

use inference_contract::*;

fn remote_candidate() -> InferenceCandidate {
    InferenceCandidate {
        provider_id: "test-provider".into(),
        model: ModelIdentity::ProviderAttested {
            provider: "test-provider".into(),
            declared_model: "test-model".into(),
        },
        location: ExecutionLocation::RemoteProvider,
        supported_purposes: vec![
            InferencePurpose::Translation,
            InferencePurpose::ScientificReasoning,
        ],
        context_window_tokens: 32_768,
        supports_streaming: true,
        supports_tools: false,
        supports_structured_output: true,
        training_policy: ProviderTrainingPolicy::Never,
        retention_policy: ProviderRetentionPolicy::ZeroRetention,
        routing_policy: ProviderRoutingPolicy::DirectOnly,
        max_charge_microusd: Some(0),
    }
}

fn request(class: InformationClass) -> InferenceRequest {
    let mut requirements =
        InferenceRequirements::minimal(InferencePurpose::Translation, class);
    requirements.estimated_input_tokens = 100;
    InferenceRequest::new("translate this", requirements)
}

#[test]
fn sovereign_default_does_not_treat_credentials_as_remote_authority() {
    let err = InferencePolicy::sovereign_default()
        .admit(&request(InformationClass::Public), &remote_candidate())
        .unwrap_err();
    assert_eq!(err, InferenceAdmissionError::RemoteExecutionForbidden);
}

#[test]
fn free_private_admits_exact_zero_retention_zero_training_direct_candidate() {
    let route = InferencePolicy::free_private()
        .admit(&request(InformationClass::UserProvided), &remote_candidate())
        .unwrap();
    assert_eq!(route.provider_id(), "test-provider");
    assert_eq!(route.location(), ExecutionLocation::RemoteProvider);
    assert_eq!(route.admitted_max_charge_microusd(), 0);
}

#[test]
fn raw_cognitive_state_is_never_remote_even_under_permissive_policy() {
    let policy = InferencePolicy {
        allow_remote: true,
        allow_provider_training: true,
        allow_provider_retention: true,
        allow_third_party_routing: true,
        max_charge_microusd: 1_000_000,
    };
    let err = policy
        .admit(&request(InformationClass::CognitiveState), &remote_candidate())
        .unwrap_err();
    assert_eq!(
        err,
        InferenceAdmissionError::RawSensitiveDataCannotLeaveDevice(
            InformationClass::CognitiveState
        )
    );
}

#[test]
fn remote_safe_derivative_requires_a_new_class_but_can_then_be_admitted() {
    let route = InferencePolicy::free_private()
        .admit(
            &request(InformationClass::RemoteSafeDerived),
            &remote_candidate(),
        )
        .unwrap();
    assert_eq!(route.information_class(), InformationClass::RemoteSafeDerived);
}

#[test]
fn unknown_or_training_permitted_provider_fails_closed_under_private_policy() {
    for training_policy in [
        ProviderTrainingPolicy::Unknown,
        ProviderTrainingPolicy::MayTrain,
    ] {
        let mut candidate = remote_candidate();
        candidate.training_policy = training_policy;
        let err = InferencePolicy::free_private()
            .admit(&request(InformationClass::Public), &candidate)
            .unwrap_err();
        assert_eq!(
            err,
            InferenceAdmissionError::ProviderTrainingPolicyRejected(training_policy)
        );
    }
}

#[test]
fn unknown_cost_fails_closed_even_when_the_declared_budget_is_zero() {
    let mut candidate = remote_candidate();
    candidate.max_charge_microusd = None;
    let err = InferencePolicy::free_private()
        .admit(&request(InformationClass::Public), &candidate)
        .unwrap_err();
    assert_eq!(err, InferenceAdmissionError::CostUnknown);
}

#[test]
fn capability_requirements_are_hard_gates_not_scoring_preferences() {
    let mut req = request(InformationClass::Public);
    req.requirements.require_tools = true;
    let err = InferencePolicy::free_private()
        .admit(&req, &remote_candidate())
        .unwrap_err();
    assert_eq!(err, InferenceAdmissionError::ToolsUnsupported);
}

#[test]
fn local_private_memory_is_admitted_by_sovereign_default() {
    let mut candidate = InferenceCandidate::local(
        "local-broca",
        ModelIdentity::ContentVerified {
            digest: "blake3:test".into(),
        },
    );
    candidate.supported_purposes = vec![InferencePurpose::Translation];
    candidate.context_window_tokens = 4_096;

    let route = InferencePolicy::sovereign_default()
        .admit(&request(InformationClass::PrivateMemory), &candidate)
        .unwrap();
    assert_eq!(route.location(), ExecutionLocation::LocalDevice);
}

#[test]
fn context_budget_includes_input_and_reserved_output() {
    let mut req = request(InformationClass::Public);
    req.requirements.estimated_input_tokens = 32_700;
    req.requirements.max_output_tokens = 100;

    let err = InferencePolicy::free_private()
        .admit(&req, &remote_candidate())
        .unwrap_err();
    assert_eq!(
        err,
        InferenceAdmissionError::ContextWindowTooSmall {
            required: 32_800,
            available: 32_768,
        }
    );
}
