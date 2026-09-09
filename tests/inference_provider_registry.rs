#[path = "../src/language/inference_contract.rs"]
mod inference_contract;
#[path = "../src/language/inference_provider_registry.rs"]
mod inference_provider_registry;

use inference_contract::{
    InferencePolicy, InferencePurpose, InferenceRequest, InferenceRequirements, InformationClass,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_provider_registry::{
    ProviderClaimEvidence, ProviderClaimSourceKind, ProviderDataPolicyClaim,
    ProviderEndpointDescriptor, ProviderModelCapabilities, ProviderModelLifecycle,
    ProviderModelProfile, ProviderQualificationError, ProviderQualificationPolicy,
    ProviderRegistry, ProviderRegistryError, ProviderRequestCostBound, SourcedProviderClaim,
    digest_provider_snapshot,
};

const NOW: u64 = 1_000;

fn evidence(
    kind: ProviderClaimSourceKind,
    label: &str,
    observed: u64,
    valid_until: u64,
) -> ProviderClaimEvidence {
    ProviderClaimEvidence::new(
        kind,
        format!("source:{label}"),
        digest_provider_snapshot(label.as_bytes()),
        observed,
        valid_until,
    )
    .unwrap()
}

fn capabilities() -> ProviderModelCapabilities {
    ProviderModelCapabilities {
        supported_purposes: vec![
            InferencePurpose::GeneralReasoning,
            InferencePurpose::Translation,
        ],
        context_window_tokens: 32_768,
        supports_streaming: true,
        supports_tools: false,
        supports_structured_output: true,
    }
}

fn data_policy() -> ProviderDataPolicyClaim {
    ProviderDataPolicyClaim {
        training_policy: ProviderTrainingPolicy::Never,
        retention_policy: ProviderRetentionPolicy::ZeroRetention,
        routing_policy: ProviderRoutingPolicy::DirectOnly,
    }
}

fn profile_with(
    epoch: u64,
    lifecycle: ProviderModelLifecycle,
    lifecycle_source: ProviderClaimSourceKind,
    capability_source: ProviderClaimSourceKind,
    policy_source: ProviderClaimSourceKind,
    cost_source: ProviderClaimSourceKind,
    policy_valid_until: u64,
) -> ProviderModelProfile {
    ProviderModelProfile::new(
        "example-provider",
        "example-deployment-a",
        "reasoner-v1",
        epoch,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible("https://example.test/v1").unwrap(),
            evidence(
                ProviderClaimSourceKind::FirstPartyModelCatalog,
                "endpoint",
                900,
                2_000,
            ),
        ),
        SourcedProviderClaim::new(
            lifecycle,
            evidence(lifecycle_source, "lifecycle", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            capabilities(),
            evidence(capability_source, "capabilities", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            data_policy(),
            evidence(policy_source, "policy", 900, policy_valid_until),
        ),
        SourcedProviderClaim::new(
            ProviderRequestCostBound::new("account-scope-a", 0).unwrap(),
            evidence(cost_source, "account-cost", 900, 2_000),
        ),
    )
    .unwrap()
}

fn production_profile(epoch: u64) -> ProviderModelProfile {
    profile_with(
        epoch,
        ProviderModelLifecycle::Production,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyPolicyDocument,
        ProviderClaimSourceKind::FirstPartyAccountState,
        2_000,
    )
}

#[test]
fn fresh_first_party_claims_materialize_executable_candidate() {
    let qualified = production_profile(1)
        .qualify(NOW, ProviderQualificationPolicy::default())
        .unwrap();

    assert_eq!(qualified.deployment_id(), "example-deployment-a");
    assert_eq!(qualified.account_scope_id(), "account-scope-a");
    assert_eq!(qualified.candidate().provider_id, "example-provider");
    assert_eq!(qualified.candidate().max_charge_microusd, Some(0));
    assert!(qualified.is_fresh_at(1_999));
    assert!(!qualified.is_fresh_at(2_000));

    // The qualified candidate still passes the existing IF-0 policy theorem.
    let request = InferenceRequest::new(
        "public question",
        InferenceRequirements::minimal(
            InferencePurpose::GeneralReasoning,
            InformationClass::RemoteSafeDerived,
        ),
    );
    InferencePolicy::free_private()
        .admit(&request, qualified.candidate())
        .unwrap();
}

#[test]
fn expiry_of_any_critical_claim_fails_closed() {
    let err = profile_with(
        1,
        ProviderModelLifecycle::Production,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyPolicyDocument,
        ProviderClaimSourceKind::FirstPartyAccountState,
        NOW,
    )
    .qualify(NOW, ProviderQualificationPolicy::default())
    .unwrap_err();

    assert!(matches!(err, ProviderQualificationError::ClaimExpired(_)));
}

#[test]
fn future_dated_claim_is_not_yet_valid() {
    let future = ProviderClaimEvidence::new(
        ProviderClaimSourceKind::FirstPartyPolicyDocument,
        "source:future-policy",
        digest_provider_snapshot(b"future-policy"),
        1_100,
        2_000,
    )
    .unwrap();
    let mut profile = production_profile(1);
    // Build a sibling profile because fields are intentionally private.
    profile = ProviderModelProfile::new(
        profile.provider_id(),
        profile.deployment_id(),
        profile.model_id(),
        2,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible("https://example.test/v1").unwrap(),
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "endpoint-2", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            ProviderModelLifecycle::Production,
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "lifecycle-2", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            capabilities(),
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "capabilities-2", 900, 2_000),
        ),
        SourcedProviderClaim::new(data_policy(), future),
        SourcedProviderClaim::new(
            ProviderRequestCostBound::new("account-scope-a", 0).unwrap(),
            evidence(ProviderClaimSourceKind::FirstPartyAccountState, "cost-2", 900, 2_000),
        ),
    )
    .unwrap();

    assert!(matches!(
        profile
            .qualify(NOW, ProviderQualificationPolicy::default())
            .unwrap_err(),
        ProviderQualificationError::ClaimNotYetValid(_)
    ));
}

#[test]
fn public_pricing_document_cannot_mint_account_cost_authority() {
    let err = profile_with(
        1,
        ProviderModelLifecycle::Production,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyPolicyDocument,
        ProviderClaimSourceKind::FirstPartyPricingDocument,
        2_000,
    )
    .qualify(NOW, ProviderQualificationPolicy::default())
    .unwrap_err();

    assert!(matches!(
        err,
        ProviderQualificationError::SourceNotAuthoritative { .. }
    ));
}

#[test]
fn model_catalog_cannot_substitute_for_privacy_policy() {
    let err = profile_with(
        1,
        ProviderModelLifecycle::Production,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyAccountState,
        2_000,
    )
    .qualify(NOW, ProviderQualificationPolicy::default())
    .unwrap_err();

    assert!(matches!(
        err,
        ProviderQualificationError::SourceNotAuthoritative { .. }
    ));
}

#[test]
fn third_party_curator_is_observable_but_not_executable_authority() {
    let err = profile_with(
        1,
        ProviderModelLifecycle::Production,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::ThirdPartyCurator,
        ProviderClaimSourceKind::FirstPartyPolicyDocument,
        ProviderClaimSourceKind::FirstPartyAccountState,
        2_000,
    )
    .qualify(NOW, ProviderQualificationPolicy::default())
    .unwrap_err();

    assert!(matches!(
        err,
        ProviderQualificationError::SourceNotAuthoritative { .. }
    ));
}

#[test]
fn preview_models_require_explicit_qualification_policy() {
    let preview = profile_with(
        1,
        ProviderModelLifecycle::Preview,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyModelCatalog,
        ProviderClaimSourceKind::FirstPartyPolicyDocument,
        ProviderClaimSourceKind::FirstPartyAccountState,
        2_000,
    );

    assert_eq!(
        preview
            .qualify(NOW, ProviderQualificationPolicy::default())
            .unwrap_err(),
        ProviderQualificationError::PreviewModelForbidden
    );
    preview
        .qualify(
            NOW,
            ProviderQualificationPolicy {
                allow_preview_models: true,
            },
        )
        .unwrap();
}

#[test]
fn registry_rejects_profile_epoch_rollback() {
    let mut registry = ProviderRegistry::default();
    registry.install(production_profile(2)).unwrap();
    assert_eq!(
        registry.install(production_profile(2)).unwrap_err(),
        ProviderRegistryError::ProfileEpochNotAdvanced
    );
    assert_eq!(
        registry.install(production_profile(1)).unwrap_err(),
        ProviderRegistryError::ProfileEpochNotAdvanced
    );
    registry.install(production_profile(3)).unwrap();
}

#[test]
fn registry_snapshot_reports_stale_profiles_instead_of_silently_dropping_them() {
    let mut registry = ProviderRegistry::default();
    registry.install(production_profile(1)).unwrap();
    let snapshot = registry.qualify_all(2_001, ProviderQualificationPolicy::default());
    assert!(snapshot.qualified().is_empty());
    assert_eq!(snapshot.rejected().len(), 1);
}

#[test]
fn semantic_set_order_does_not_change_profile_digest() {
    let first = production_profile(1)
        .qualify(NOW, ProviderQualificationPolicy::default())
        .unwrap();

    let alternate_caps = ProviderModelCapabilities {
        supported_purposes: vec![
            InferencePurpose::Translation,
            InferencePurpose::GeneralReasoning,
            InferencePurpose::Translation,
        ],
        ..capabilities()
    };
    let second_profile = ProviderModelProfile::new(
        "example-provider",
        "example-deployment-a",
        "reasoner-v1",
        1,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible("https://example.test/v1").unwrap(),
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "endpoint", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            ProviderModelLifecycle::Production,
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "lifecycle", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            alternate_caps,
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "capabilities", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            data_policy(),
            evidence(ProviderClaimSourceKind::FirstPartyPolicyDocument, "policy", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            ProviderRequestCostBound::new("account-scope-a", 0).unwrap(),
            evidence(ProviderClaimSourceKind::FirstPartyAccountState, "account-cost", 900, 2_000),
        ),
    )
    .unwrap();
    let second = second_profile
        .qualify(NOW, ProviderQualificationPolicy::default())
        .unwrap();

    assert_eq!(first.profile_digest(), second.profile_digest());
}

#[test]
fn provenance_snapshot_change_changes_profile_digest_even_when_semantics_match() {
    let first = production_profile(1)
        .qualify(NOW, ProviderQualificationPolicy::default())
        .unwrap();

    let changed = ProviderModelProfile::new(
        "example-provider",
        "example-deployment-a",
        "reasoner-v1",
        1,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible("https://example.test/v1").unwrap(),
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "endpoint", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            ProviderModelLifecycle::Production,
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "lifecycle", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            capabilities(),
            evidence(ProviderClaimSourceKind::FirstPartyModelCatalog, "capabilities-new-snapshot", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            data_policy(),
            evidence(ProviderClaimSourceKind::FirstPartyPolicyDocument, "policy", 900, 2_000),
        ),
        SourcedProviderClaim::new(
            ProviderRequestCostBound::new("account-scope-a", 0).unwrap(),
            evidence(ProviderClaimSourceKind::FirstPartyAccountState, "account-cost", 900, 2_000),
        ),
    )
    .unwrap()
    .qualify(NOW, ProviderQualificationPolicy::default())
    .unwrap();

    assert_ne!(first.profile_digest(), changed.profile_digest());
}

#[test]
fn evidence_debug_omits_source_locator() {
    let claim = evidence(
        ProviderClaimSourceKind::FirstPartyAccountState,
        "account-sensitive-locator",
        900,
        2_000,
    );
    let debug = format!("{claim:?}");
    assert!(!debug.contains("account-sensitive-locator"));
    assert!(debug.contains("FirstPartyAccountState"));
}
