#[path = "../src/language/inference_contract.rs"]
mod inference_contract;
#[path = "../src/language/inference_permit.rs"]
mod inference_permit;
#[path = "../src/language/inference_binding.rs"]
mod inference_binding;
#[path = "../src/language/inference_provider_registry.rs"]
mod inference_provider_registry;
#[path = "../src/language/inference_resource_scope.rs"]
mod inference_resource_scope;

use inference_binding::{CredentialStateBinding, QuotaStateBinding};
use inference_contract::{
    InferencePurpose, ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_provider_registry::{
    ProviderClaimEvidence, ProviderClaimSourceKind, ProviderDataPolicyClaim,
    ProviderEndpointDescriptor, ProviderModelCapabilities, ProviderModelLifecycle,
    ProviderModelProfile, ProviderQualificationPolicy, ProviderRequestCostBound,
    QualifiedProviderCandidate, SourcedProviderClaim, digest_provider_snapshot,
};
use inference_resource_scope::{
    InferenceResourceScopeError, InferenceResourceScopeKey, InferenceResourceScopeRegistry,
    LocalScopeRegistrationEvidence,
};

const PROVIDER: &str = "example-provider";
const DEPLOYMENT: &str = "deployment-a";
const ACCOUNT: &str = "account-a";
const MODEL: &str = "reasoner-v1";

fn provider_evidence(kind: ProviderClaimSourceKind, label: &str) -> ProviderClaimEvidence {
    ProviderClaimEvidence::new(
        kind,
        format!("source:{label}"),
        digest_provider_snapshot(label.as_bytes()),
        90,
        500,
    )
    .unwrap()
}

fn qualified_profile(profile_epoch: u64, salt: &str) -> QualifiedProviderCandidate {
    ProviderModelProfile::new(
        PROVIDER,
        DEPLOYMENT,
        MODEL,
        profile_epoch,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible("https://example.test/v1").unwrap(),
            provider_evidence(
                ProviderClaimSourceKind::FirstPartyModelCatalog,
                &format!("endpoint:{salt}"),
            ),
        ),
        SourcedProviderClaim::new(
            ProviderModelLifecycle::Production,
            provider_evidence(
                ProviderClaimSourceKind::FirstPartyModelCatalog,
                &format!("lifecycle:{salt}"),
            ),
        ),
        SourcedProviderClaim::new(
            ProviderModelCapabilities {
                supported_purposes: vec![InferencePurpose::GeneralReasoning],
                context_window_tokens: 32_768,
                supports_streaming: true,
                supports_tools: false,
                supports_structured_output: false,
            },
            provider_evidence(
                ProviderClaimSourceKind::FirstPartyModelCatalog,
                &format!("capabilities:{salt}"),
            ),
        ),
        SourcedProviderClaim::new(
            ProviderDataPolicyClaim {
                training_policy: ProviderTrainingPolicy::Never,
                retention_policy: ProviderRetentionPolicy::ZeroRetention,
                routing_policy: ProviderRoutingPolicy::DirectOnly,
            },
            provider_evidence(
                ProviderClaimSourceKind::FirstPartyPolicyDocument,
                &format!("policy:{salt}"),
            ),
        ),
        SourcedProviderClaim::new(
            ProviderRequestCostBound::new(ACCOUNT, 0).unwrap(),
            provider_evidence(
                ProviderClaimSourceKind::FirstPartyAccountState,
                &format!("cost:{salt}"),
            ),
        ),
    )
    .unwrap()
    .qualify(100, ProviderQualificationPolicy::default())
    .unwrap()
}

fn credential(epoch: u64) -> CredentialStateBinding {
    CredentialStateBinding::new("credential-a", epoch).unwrap()
}

fn quota(epoch: u64) -> QuotaStateBinding {
    QuotaStateBinding::new("quota-a", epoch).unwrap()
}

fn scope() -> InferenceResourceScopeKey {
    InferenceResourceScopeKey::new(PROVIDER, DEPLOYMENT, ACCOUNT).unwrap()
}

fn local_evidence(epoch: u64, label: &str) -> LocalScopeRegistrationEvidence {
    LocalScopeRegistrationEvidence::new(epoch, *blake3::hash(label.as_bytes()).as_bytes()).unwrap()
}

fn coherent_registry(
    registry_id: [u8; 32],
    credential_binding: &CredentialStateBinding,
    quota_binding: &QuotaStateBinding,
    credential_label: &str,
    quota_label: &str,
) -> InferenceResourceScopeRegistry {
    let mut registry = InferenceResourceScopeRegistry::new(registry_id).unwrap();
    registry
        .register_credential(
            credential_binding,
            scope(),
            local_evidence(1, credential_label),
        )
        .unwrap();
    registry
        .register_quota(quota_binding, scope(), local_evidence(1, quota_label))
        .unwrap();
    registry
}

#[test]
fn coherent_explicit_scope_registration_verifies() {
    let profile = qualified_profile(1, "profile-a");
    let credential = credential(1);
    let quota = quota(1);
    let registry = coherent_registry([1; 32], &credential, &quota, "cred-a", "quota-a");

    let verified = registry.verify(&profile, &credential, &quota).unwrap();
    assert_eq!(verified.scope().provider_id(), PROVIDER);
    assert_eq!(verified.scope().deployment_id(), DEPLOYMENT);
    assert_eq!(verified.scope().account_scope_id(), ACCOUNT);
    assert_eq!(verified.credential_id(), "credential-a");
    assert_eq!(verified.quota_scope_id(), "quota-a");
    assert_ne!(verified.proof_digest().as_bytes(), &[0; 32]);
}

#[test]
fn matching_strings_do_not_create_scope_authority() {
    let profile = qualified_profile(1, "profile-a");
    let credential = CredentialStateBinding::new("example-provider:account-a", 1).unwrap();
    let quota = QuotaStateBinding::new("example-provider:account-a", 1).unwrap();
    let registry = InferenceResourceScopeRegistry::new([2; 32]).unwrap();

    assert_eq!(
        registry.verify(&profile, &credential, &quota).unwrap_err(),
        InferenceResourceScopeError::CredentialRegistrationMissing
    );
}

#[test]
fn credential_registered_to_another_account_is_rejected() {
    let profile = qualified_profile(1, "profile-a");
    let credential = credential(1);
    let quota = quota(1);
    let mut registry = InferenceResourceScopeRegistry::new([3; 32]).unwrap();
    let wrong = InferenceResourceScopeKey::new(PROVIDER, DEPLOYMENT, "account-b").unwrap();
    registry
        .register_credential(&credential, wrong, local_evidence(1, "cred-wrong"))
        .unwrap();
    registry
        .register_quota(&quota, scope(), local_evidence(1, "quota-right"))
        .unwrap();

    assert_eq!(
        registry.verify(&profile, &credential, &quota).unwrap_err(),
        InferenceResourceScopeError::CredentialScopeMismatch
    );
}

#[test]
fn quota_registered_to_another_deployment_is_rejected() {
    let profile = qualified_profile(1, "profile-a");
    let credential = credential(1);
    let quota = quota(1);
    let mut registry = InferenceResourceScopeRegistry::new([4; 32]).unwrap();
    registry
        .register_credential(&credential, scope(), local_evidence(1, "cred-right"))
        .unwrap();
    let wrong = InferenceResourceScopeKey::new(PROVIDER, "deployment-b", ACCOUNT).unwrap();
    registry
        .register_quota(&quota, wrong, local_evidence(1, "quota-wrong"))
        .unwrap();

    assert_eq!(
        registry.verify(&profile, &credential, &quota).unwrap_err(),
        InferenceResourceScopeError::QuotaScopeMismatch
    );
}

#[test]
fn newer_credential_epoch_supersedes_old_scope_registration() {
    let profile = qualified_profile(1, "profile-a");
    let credential_v1 = credential(1);
    let credential_v2 = credential(2);
    let quota = quota(1);
    let mut registry = coherent_registry([5; 32], &credential_v1, &quota, "cred-v1", "quota");
    registry
        .register_credential(&credential_v2, scope(), local_evidence(2, "cred-v2"))
        .unwrap();

    assert_eq!(
        registry
            .verify(&profile, &credential_v1, &quota)
            .unwrap_err(),
        InferenceResourceScopeError::CredentialRegistrationSuperseded
    );
    registry.verify(&profile, &credential_v2, &quota).unwrap();
}

#[test]
fn newer_quota_epoch_supersedes_old_scope_registration() {
    let profile = qualified_profile(1, "profile-a");
    let credential = credential(1);
    let quota_v1 = quota(1);
    let quota_v2 = quota(2);
    let mut registry = coherent_registry([6; 32], &credential, &quota_v1, "cred", "quota-v1");
    registry
        .register_quota(&quota_v2, scope(), local_evidence(2, "quota-v2"))
        .unwrap();

    assert_eq!(
        registry
            .verify(&profile, &credential, &quota_v1)
            .unwrap_err(),
        InferenceResourceScopeError::QuotaRegistrationSuperseded
    );
    registry.verify(&profile, &credential, &quota_v2).unwrap();
}

#[test]
fn registration_epoch_rollback_is_rejected() {
    let credential_v2 = credential(2);
    let credential_v1 = credential(1);
    let quota = quota(1);
    let mut registry = InferenceResourceScopeRegistry::new([7; 32]).unwrap();
    registry
        .register_credential(&credential_v2, scope(), local_evidence(2, "cred-v2"))
        .unwrap();
    assert_eq!(
        registry
            .register_credential(&credential_v1, scope(), local_evidence(3, "cred-v1"))
            .unwrap_err(),
        InferenceResourceScopeError::CredentialEpochNotAdvanced
    );

    registry
        .register_quota(&quota, scope(), local_evidence(1, "quota"))
        .unwrap();
    assert_eq!(
        registry
            .register_quota(&quota, scope(), local_evidence(2, "quota-repeat"))
            .unwrap_err(),
        InferenceResourceScopeError::QuotaEpochNotAdvanced
    );
}

#[test]
fn registry_lineage_and_registration_evidence_are_part_of_proof() {
    let profile = qualified_profile(1, "profile-a");
    let credential = credential(1);
    let quota = quota(1);

    let first = coherent_registry([8; 32], &credential, &quota, "cred-a", "quota-a")
        .verify(&profile, &credential, &quota)
        .unwrap();
    let changed_source = coherent_registry([8; 32], &credential, &quota, "cred-b", "quota-a")
        .verify(&profile, &credential, &quota)
        .unwrap();
    let changed_registry = coherent_registry([9; 32], &credential, &quota, "cred-a", "quota-a")
        .verify(&profile, &credential, &quota)
        .unwrap();

    assert_ne!(first.proof_digest(), changed_source.proof_digest());
    assert_ne!(first.proof_digest(), changed_registry.proof_digest());
}

#[test]
fn provider_profile_snapshot_is_part_of_resource_scope_proof() {
    let profile_a = qualified_profile(1, "profile-a");
    let profile_b = qualified_profile(2, "profile-b");
    let credential = credential(1);
    let quota = quota(1);
    let registry = coherent_registry([10; 32], &credential, &quota, "cred", "quota");

    let first = registry.verify(&profile_a, &credential, &quota).unwrap();
    let second = registry.verify(&profile_b, &credential, &quota).unwrap();
    assert_ne!(first.proof_digest(), second.proof_digest());
}

#[test]
fn zero_registry_identity_fails_closed() {
    assert_eq!(
        InferenceResourceScopeRegistry::new([0; 32]).unwrap_err(),
        InferenceResourceScopeError::ZeroRegistryId
    );
}
