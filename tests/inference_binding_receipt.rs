#[path = "../src/language/inference_contract.rs"]
mod inference_contract;
#[path = "../src/language/inference_permit.rs"]
mod inference_permit;
#[path = "../src/language/inference_binding.rs"]
mod inference_binding;
#[path = "../src/language/inference_receipt.rs"]
mod inference_receipt;

use inference_binding::{
    CredentialStateBinding, QuotaStateBinding, admit_and_bind, digest_candidate,
    digest_credential_state, digest_quota_state, digest_request,
};
use inference_contract::{
    ExecutionLocation, InferenceCandidate, InferencePolicy, InferencePurpose, InferenceRequest,
    InferenceRequirements, InformationClass, ModelIdentity, ProviderRetentionPolicy,
    ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_permit::InferencePermitIssuer;
use inference_receipt::{InferenceCompletion, InferenceFailureClass, InferenceReceipt};

fn request(prompt: &str) -> InferenceRequest {
    let mut requirements = InferenceRequirements::minimal(
        InferencePurpose::GeneralReasoning,
        InformationClass::RemoteSafeDerived,
    );
    requirements.estimated_input_tokens = 128;
    requirements.max_output_tokens = 256;
    InferenceRequest::new(prompt, requirements)
}

fn candidate() -> InferenceCandidate {
    InferenceCandidate {
        provider_id: "example-provider".to_string(),
        model: ModelIdentity::ProviderAttested {
            provider: "example-provider".to_string(),
            declared_model: "reasoner-v1".to_string(),
        },
        location: ExecutionLocation::RemoteProvider,
        supported_purposes: vec![InferencePurpose::GeneralReasoning],
        context_window_tokens: 4096,
        supports_streaming: true,
        supports_tools: false,
        supports_structured_output: false,
        training_policy: ProviderTrainingPolicy::Never,
        retention_policy: ProviderRetentionPolicy::ZeroRetention,
        routing_policy: ProviderRoutingPolicy::DirectOnly,
        max_charge_microusd: Some(0),
    }
}

fn credential(epoch: u64) -> CredentialStateBinding {
    CredentialStateBinding::new("credential:example-provider:primary", epoch).unwrap()
}

fn quota(epoch: u64, remaining: u64) -> QuotaStateBinding {
    QuotaStateBinding::new("quota:example-provider:project-a", epoch)
        .unwrap()
        .with_remaining_requests(Some(remaining))
        .with_remaining_tokens(Some(100_000))
        .with_reset_at_tick(Some(1_000))
}

#[test]
fn identical_semantics_produce_identical_binding() {
    let policy = InferencePolicy::free_private();
    let first = admit_and_bind(&policy, &request("hello"), &candidate(), &credential(1), &quota(1, 50))
        .unwrap();
    let second = admit_and_bind(&policy, &request("hello"), &candidate(), &credential(1), &quota(1, 50))
        .unwrap();
    assert_eq!(first.binding(), second.binding());
}

#[test]
fn prompt_change_changes_only_request_digest() {
    let policy = InferencePolicy::free_private();
    let first = admit_and_bind(&policy, &request("alpha"), &candidate(), &credential(1), &quota(1, 50))
        .unwrap();
    let second = admit_and_bind(&policy, &request("beta"), &candidate(), &credential(1), &quota(1, 50))
        .unwrap();

    assert_ne!(first.binding().request_digest, second.binding().request_digest);
    assert_eq!(first.binding().route_digest, second.binding().route_digest);
    assert_eq!(first.binding().policy_digest, second.binding().policy_digest);
    assert_eq!(first.binding().provider_state_digest, second.binding().provider_state_digest);
    assert_eq!(first.binding().credential_state_digest, second.binding().credential_state_digest);
    assert_eq!(first.binding().quota_state_digest, second.binding().quota_state_digest);
}

#[test]
fn provider_privacy_drift_changes_provider_digest() {
    let mut changed = candidate();
    let original = digest_candidate(&changed).unwrap();
    changed.retention_policy = ProviderRetentionPolicy::MayRetain;
    let drifted = digest_candidate(&changed).unwrap();
    assert_ne!(original, drifted);
}

#[test]
fn purpose_set_order_and_duplicates_do_not_change_provider_digest() {
    let mut first = candidate();
    first.supported_purposes = vec![
        InferencePurpose::GeneralReasoning,
        InferencePurpose::Translation,
        InferencePurpose::GeneralReasoning,
    ];
    let mut second = candidate();
    second.supported_purposes = vec![
        InferencePurpose::Translation,
        InferencePurpose::GeneralReasoning,
    ];
    assert_eq!(digest_candidate(&first).unwrap(), digest_candidate(&second).unwrap());
}

#[test]
fn credential_and_quota_epochs_are_independent_bindings() {
    assert_ne!(
        digest_credential_state(&credential(1)).unwrap(),
        digest_credential_state(&credential(2)).unwrap()
    );
    assert_ne!(
        digest_quota_state(&quota(1, 50)).unwrap(),
        digest_quota_state(&quota(2, 50)).unwrap()
    );
    assert_ne!(
        digest_quota_state(&quota(1, 50)).unwrap(),
        digest_quota_state(&quota(1, 49)).unwrap()
    );
}

#[test]
fn request_digest_covers_system_prompt_and_requirements() {
    let mut first = request("same");
    first.system_prompt = Some("system-a".to_string());
    let mut second = request("same");
    second.system_prompt = Some("system-b".to_string());
    assert_ne!(digest_request(&first).unwrap(), digest_request(&second).unwrap());

    let mut third = request("same");
    third.requirements.require_tools = true;
    assert_ne!(digest_request(&request("same")).unwrap(), digest_request(&third).unwrap());
}

#[test]
fn raw_cognitive_state_cannot_reach_bound_remote_admission() {
    let mut req = request("private internal state");
    req.requirements.information_class = InformationClass::CognitiveState;
    let err = admit_and_bind(
        &InferencePolicy {
            allow_remote: true,
            allow_provider_training: true,
            allow_provider_retention: true,
            allow_third_party_routing: true,
            max_charge_microusd: u64::MAX,
        },
        &req,
        &candidate(),
        &credential(1),
        &quota(1, 50),
    )
    .unwrap_err();
    assert!(err.to_string().contains("cannot cross a remote inference boundary"));
}

fn prepared() -> inference_permit::PreparedInferenceExecution {
    let bound = admit_and_bind(
        &InferencePolicy::free_private(),
        &request("receipt test"),
        &candidate(),
        &credential(1),
        &quota(1, 50),
    )
    .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 10, [42; 32]).unwrap();
    issuer.prepare_execution(permit, binding, 101).unwrap()
}

#[test]
fn success_receipt_binds_output_without_retaining_raw_text() {
    let receipt = InferenceReceipt::success(prepared(), "sensitive model response", 102).unwrap();
    assert!(matches!(receipt.completion(), InferenceCompletion::Success { .. }));
    assert!(receipt.response_digest().is_some());
    let debug = format!("{receipt:?}");
    assert!(!debug.contains("sensitive model response"));
    assert_eq!(receipt.prepared_at_tick(), 101);
    assert_eq!(receipt.completed_at_tick(), 102);
}

#[test]
fn different_outputs_have_different_receipt_digests() {
    let first = InferenceReceipt::success(prepared(), "alpha", 102).unwrap();
    let second = InferenceReceipt::success(prepared(), "beta", 102).unwrap();
    assert_ne!(first.response_digest(), second.response_digest());
}

#[test]
fn failure_receipt_has_no_response_digest() {
    let receipt = InferenceReceipt::failure(prepared(), InferenceFailureClass::Transport, 102).unwrap();
    assert!(matches!(receipt.completion(), InferenceCompletion::Failure { .. }));
    assert!(receipt.response_digest().is_none());
}

#[test]
fn receipt_rejects_completion_before_preparation() {
    let err = InferenceReceipt::failure(prepared(), InferenceFailureClass::Cancelled, 100).unwrap_err();
    assert!(err.to_string().contains("precedes execution preparation"));
}
