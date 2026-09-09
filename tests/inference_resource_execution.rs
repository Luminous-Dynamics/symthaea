#[path = "../src/language/inference_contract.rs"]
mod inference_contract;
#[path = "../src/language/inference_permit.rs"]
mod inference_permit;
#[path = "../src/language/inference_binding.rs"]
mod inference_binding;
#[path = "../src/language/inference_receipt.rs"]
mod inference_receipt;
#[path = "../src/language/openai_compatible_transport.rs"]
mod openai_compatible_transport;
#[path = "../src/language/inference_execution_envelope.rs"]
mod inference_execution_envelope;
#[path = "../src/language/inference_provider_registry.rs"]
mod inference_provider_registry;
#[path = "../src/language/inference_profile_execution.rs"]
mod inference_profile_execution;
#[path = "../src/language/inference_resource_scope.rs"]
mod inference_resource_scope;
#[path = "../src/language/inference_resource_execution.rs"]
mod inference_resource_execution;
#[path = "../src/language/inference_executor.rs"]
mod inference_executor;
#[path = "../src/language/inference_credential.rs"]
mod inference_credential;
#[path = "../src/language/inference_current_profile_executor.rs"]
mod inference_current_profile_executor;

use inference_binding::{CredentialStateBinding, QuotaStateBinding};
use inference_contract::{
    InferencePolicy, InferencePurpose, InferenceRequest, InferenceRequirements, InformationClass,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_credential::InferenceCredentialLease;
use inference_current_profile_executor::CurrentProfileCredentialExecutor;
use inference_execution_envelope::InferenceGenerationControls;
use inference_executor::{InferenceExecutionFailure, InferenceTickSource};
use inference_permit::InferencePermitIssuer;
use inference_profile_execution::{ProviderProfileResolveError, ProviderProfileResolver};
use inference_provider_registry::{
    ProviderClaimEvidence, ProviderClaimSourceKind, ProviderDataPolicyClaim,
    ProviderEndpointDescriptor, ProviderModelCapabilities, ProviderModelLifecycle,
    ProviderModelProfile, ProviderProfileKey, ProviderQualificationPolicy, ProviderRegistry,
    ProviderRequestCostBound, QualifiedProviderCandidate, SourcedProviderClaim,
    digest_provider_snapshot,
};
use inference_resource_execution::{
    InferenceResourceExecutionError, InferenceResourceScopeResolver,
    ResourceScopedCurrentProfileExecutor,
};
use inference_resource_scope::{
    InferenceResourceScopeError, InferenceResourceScopeKey, InferenceResourceScopeRegistry,
    LocalScopeRegistrationEvidence, VerifiedInferenceResourceScope,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const PROVIDER: &str = "example-provider";
const DEPLOYMENT: &str = "deployment-a";
const ACCOUNT: &str = "account-a";
const MODEL: &str = "reasoner-v1";
const CREDENTIAL_ID: &str = "credential-a";
const QUOTA_ID: &str = "quota-a";

#[derive(Debug)]
struct SharedTick(AtomicU64);

impl SharedTick {
    fn new(tick: u64) -> Self {
        Self(AtomicU64::new(tick))
    }

    fn set(&self, tick: u64) {
        self.0.store(tick, Ordering::SeqCst);
    }
}

impl InferenceTickSource for SharedTick {
    fn now_tick(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Default)]
struct SharedProviderRegistry(Arc<Mutex<ProviderRegistry>>);

impl SharedProviderRegistry {
    fn install(&self, profile: ProviderModelProfile) {
        self.0.lock().unwrap().install(profile).unwrap();
    }
}

impl ProviderProfileResolver for SharedProviderRegistry {
    fn resolve_current(
        &self,
        key: &ProviderProfileKey,
        now_tick: u64,
        policy: ProviderQualificationPolicy,
    ) -> Result<QualifiedProviderCandidate, ProviderProfileResolveError> {
        let guard = self.0.lock().unwrap();
        ProviderProfileResolver::resolve_current(&*guard, key, now_tick, policy)
    }
}

#[derive(Clone)]
struct SharedScopeRegistry(Arc<Mutex<InferenceResourceScopeRegistry>>);

impl SharedScopeRegistry {
    fn new(registry_id: [u8; 32]) -> Self {
        Self(Arc::new(Mutex::new(
            InferenceResourceScopeRegistry::new(registry_id).unwrap(),
        )))
    }

    fn register_credential(&self, credential: &CredentialStateBinding, authority_epoch: u64) {
        self.0
            .lock()
            .unwrap()
            .register_credential(
                credential,
                scope(),
                local_evidence(authority_epoch, &format!("credential-{authority_epoch}")),
            )
            .unwrap();
    }

    fn register_quota(&self, quota: &QuotaStateBinding, authority_epoch: u64) {
        self.0
            .lock()
            .unwrap()
            .register_quota(
                quota,
                scope(),
                local_evidence(authority_epoch, &format!("quota-{authority_epoch}")),
            )
            .unwrap();
    }
}

impl InferenceResourceScopeResolver for SharedScopeRegistry {
    fn verify_current_scope(
        &self,
        profile: &QualifiedProviderCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
    ) -> Result<VerifiedInferenceResourceScope, InferenceResourceScopeError> {
        self.0.lock().unwrap().verify(profile, credential, quota)
    }
}

fn key() -> ProviderProfileKey {
    ProviderProfileKey {
        provider_id: PROVIDER.to_owned(),
        deployment_id: DEPLOYMENT.to_owned(),
        model_id: MODEL.to_owned(),
    }
}

fn scope() -> InferenceResourceScopeKey {
    InferenceResourceScopeKey::new(PROVIDER, DEPLOYMENT, ACCOUNT).unwrap()
}

fn local_evidence(authority_epoch: u64, label: &str) -> LocalScopeRegistrationEvidence {
    LocalScopeRegistrationEvidence::new(
        authority_epoch,
        *blake3::hash(label.as_bytes()).as_bytes(),
    )
    .unwrap()
}

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

fn profile(base_url: &str, epoch: u64, salt: &str) -> ProviderModelProfile {
    ProviderModelProfile::new(
        PROVIDER,
        DEPLOYMENT,
        MODEL,
        epoch,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible(base_url).unwrap(),
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
}

fn request() -> InferenceRequest {
    let mut requirements = InferenceRequirements::minimal(
        InferencePurpose::GeneralReasoning,
        InformationClass::RemoteSafeDerived,
    );
    requirements.estimated_input_tokens = 64;
    requirements.max_output_tokens = 128;
    InferenceRequest::new("explain the resource scope invariant", requirements)
}

fn quota(epoch: u64) -> QuotaStateBinding {
    QuotaStateBinding::new(QUOTA_ID, epoch)
        .unwrap()
        .with_remaining_requests(Some(10))
        .with_remaining_tokens(Some(10_000))
}

fn credential(epoch: u64) -> CredentialStateBinding {
    CredentialStateBinding::new(CREDENTIAL_ID, epoch).unwrap()
}

fn make_scoped_executor(
    provider_registry: SharedProviderRegistry,
    scope_registry: SharedScopeRegistry,
) -> (
    ResourceScopedCurrentProfileExecutor<SharedTick, SharedProviderRegistry, SharedScopeRegistry>,
    Arc<SharedTick>,
) {
    let clock = Arc::new(SharedTick::new(100));
    let lease = InferenceCredentialLease::anonymous(CREDENTIAL_ID, 1).unwrap();
    let current = CurrentProfileCredentialExecutor::from_current_profile(
        provider_registry,
        key(),
        ProviderQualificationPolicy::default(),
        lease,
        Duration::from_secs(2),
        1,
        clock.clone(),
    )
    .unwrap();
    (
        ResourceScopedCurrentProfileExecutor::new(current, scope_registry),
        clock,
    )
}

fn prepare_v4(
    executor: &ResourceScopedCurrentProfileExecutor<
        SharedTick,
        SharedProviderRegistry,
        SharedScopeRegistry,
    >,
    quota_binding: &QuotaStateBinding,
) -> inference_permit::PreparedInferenceExecution {
    let bound = executor
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            quota_binding,
            InferenceGenerationControls::default(),
        )
        .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 20, [113; 32]).unwrap();
    issuer.prepare_execution(permit, binding, 101).unwrap()
}

fn coherent_scope_registry(registry_id: [u8; 32]) -> SharedScopeRegistry {
    let registry = SharedScopeRegistry::new(registry_id);
    registry.register_credential(&credential(1), 1);
    registry.register_quota(&quota(1), 1);
    registry
}

#[test]
fn unchanged_scope_reverification_reproduces_same_v4_binding() {
    let providers = SharedProviderRegistry::default();
    providers.install(profile("https://example.test/v1", 1, "stable"));
    let scopes = coherent_scope_registry([1; 32]);
    let (executor, clock) = make_scoped_executor(providers, scopes);
    let quota = quota(1);

    let first = executor
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .unwrap();
    clock.set(102);
    let second = executor
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .unwrap();

    assert_eq!(first.binding(), second.binding());
}

#[test]
fn missing_scope_authority_fails_before_permit_construction() {
    let providers = SharedProviderRegistry::default();
    providers.install(profile("https://example.test/v1", 1, "missing"));
    let scopes = SharedScopeRegistry::new([2; 32]);
    let (executor, _) = make_scoped_executor(providers, scopes);

    let error = executor
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota(1),
            InferenceGenerationControls::default(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        InferenceResourceExecutionError::Scope(
            InferenceResourceScopeError::CredentialRegistrationMissing
        )
    ));
}

#[tokio::test]
async fn newer_credential_scope_epoch_invalidates_prepared_authority_before_network() {
    let dead_url = "http://127.0.0.1:9/v1";
    let providers = SharedProviderRegistry::default();
    providers.install(profile(dead_url, 1, "credential-rotation"));
    let scopes = coherent_scope_registry([3; 32]);
    let (executor, clock) = make_scoped_executor(providers, scopes.clone());
    let quota = quota(1);
    let prepared = prepare_v4(&executor, &quota);

    scopes.register_credential(&credential(2), 2);
    clock.set(102);

    let outcome = executor
        .execute_current(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        outcome.failure,
        Some(InferenceExecutionFailure::ResourceScopeRejected)
    );
    assert!(outcome.response_text.is_none());
    assert!(outcome.wire_observation.is_none());
}

#[tokio::test]
async fn newer_quota_scope_epoch_invalidates_old_quota_before_network() {
    let dead_url = "http://127.0.0.1:9/v1";
    let providers = SharedProviderRegistry::default();
    providers.install(profile(dead_url, 1, "quota-rotation"));
    let scopes = coherent_scope_registry([4; 32]);
    let (executor, clock) = make_scoped_executor(providers, scopes.clone());
    let quota_v1 = quota(1);
    let prepared = prepare_v4(&executor, &quota_v1);

    scopes.register_quota(&quota(2), 2);
    clock.set(102);

    let outcome = executor
        .execute_current(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &quota_v1,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        outcome.failure,
        Some(InferenceExecutionFailure::ResourceScopeRejected)
    );
    assert!(outcome.response_text.is_none());
    assert!(outcome.wire_observation.is_none());
}

#[test]
fn resource_registry_lineage_changes_v4_provider_state() {
    let providers_a = SharedProviderRegistry::default();
    providers_a.install(profile("https://example.test/v1", 1, "lineage"));
    let providers_b = SharedProviderRegistry::default();
    providers_b.install(profile("https://example.test/v1", 1, "lineage"));

    let (first, _) = make_scoped_executor(providers_a, coherent_scope_registry([5; 32]));
    let (second, _) = make_scoped_executor(providers_b, coherent_scope_registry([6; 32]));
    let quota = quota(1);

    let first_binding = first
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .unwrap();
    let second_binding = second
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .unwrap();

    assert_ne!(
        first_binding.binding().provider_state_digest,
        second_binding.binding().provider_state_digest
    );
}

#[tokio::test]
async fn coherent_current_resource_scope_executes_real_http_and_receipts() {
    let (base_url, captured) = spawn_one_shot_openai_server("resource-qualified response").await;
    let providers = SharedProviderRegistry::default();
    providers.install(profile(&base_url, 1, "live"));
    let scopes = coherent_scope_registry([7; 32]);
    let (executor, clock) = make_scoped_executor(providers, scopes);
    let quota = quota(1);
    let prepared = prepare_v4(&executor, &quota);
    clock.set(102);

    let outcome = executor
        .execute_current(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.failure, None);
    assert_eq!(
        outcome.response_text.as_deref(),
        Some("resource-qualified response")
    );
    assert!(outcome.receipt.response_digest().is_some());

    let request_text = captured.await.unwrap();
    assert!(request_text.contains("explain the resource scope invariant"));
    assert!(request_text.contains("\"model\":\"reasoner-v1\""));
}

async fn spawn_one_shot_openai_server(
    response_text: &'static str,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let response_body = format!(
        "{{\"id\":\"resp-resource\",\"model\":\"reasoner-v1\",\"choices\":[{{\"finish_reason\":\"stop\",\"message\":{{\"content\":\"{}\"}}}}]}}",
        response_text
    );
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut scratch = [0_u8; 4096];
        let mut expected_total = None;
        loop {
            let count = stream.read(&mut scratch).await.unwrap();
            if count == 0 {
                break;
            }
            request.extend_from_slice(&scratch[..count]);
            if expected_total.is_none()
                && let Some(header_end) = find_header_end(&request)
            {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                expected_total = Some(header_end + 4 + content_length);
            }
            if expected_total.is_some_and(|total| request.len() >= total) {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8_lossy(&request).to_string()
    });
    (format!("http://{address}/v1"), handle)
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}
