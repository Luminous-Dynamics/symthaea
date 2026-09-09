#[path = "../src/language/inference_contract.rs"]
mod inference_contract;
#[path = "../src/language/inference_provider_registry.rs"]
mod inference_provider_registry;

use inference_contract::{
    InferencePurpose, ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_provider_registry::{
    ProviderClaimEvidence, ProviderClaimSourceKind, ProviderDataPolicyClaim,
    ProviderEndpointDescriptor, ProviderModelCapabilities, ProviderModelLifecycle,
    ProviderModelProfile, ProviderQualificationError, ProviderQualificationPolicy,
    ProviderRequestCostBound, SourcedProviderClaim, digest_provider_snapshot,
};

fn manifest_evidence(label: &str) -> ProviderClaimEvidence {
    ProviderClaimEvidence::new(
        ProviderClaimSourceKind::SignedDeploymentManifest,
        format!("manifest:{label}"),
        digest_provider_snapshot(label.as_bytes()),
        100,
        1_000,
    )
    .unwrap()
}

#[test]
fn caller_selected_signed_manifest_label_is_not_executable_authority() {
    let profile = ProviderModelProfile::new(
        "example-provider",
        "deployment-a",
        "reasoner-v1",
        1,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible("https://example.test/v1").unwrap(),
            manifest_evidence("endpoint"),
        ),
        SourcedProviderClaim::new(
            ProviderModelLifecycle::Production,
            manifest_evidence("lifecycle"),
        ),
        SourcedProviderClaim::new(
            ProviderModelCapabilities {
                supported_purposes: vec![InferencePurpose::GeneralReasoning],
                context_window_tokens: 8_192,
                supports_streaming: true,
                supports_tools: false,
                supports_structured_output: false,
            },
            manifest_evidence("capabilities"),
        ),
        SourcedProviderClaim::new(
            ProviderDataPolicyClaim {
                training_policy: ProviderTrainingPolicy::Never,
                retention_policy: ProviderRetentionPolicy::ZeroRetention,
                routing_policy: ProviderRoutingPolicy::DirectOnly,
            },
            manifest_evidence("policy"),
        ),
        SourcedProviderClaim::new(
            ProviderRequestCostBound::new("account-a", 0).unwrap(),
            manifest_evidence("cost"),
        ),
    )
    .unwrap();

    assert!(matches!(
        profile
            .qualify(200, ProviderQualificationPolicy::default())
            .unwrap_err(),
        ProviderQualificationError::SourceNotAuthoritative {
            source: ProviderClaimSourceKind::SignedDeploymentManifest,
            ..
        }
    ));
}
