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
#[path = "../src/language/inference_executor.rs"]
mod inference_executor;
#[path = "../src/language/inference_credential.rs"]
mod inference_credential;
#[path = "../src/language/inference_current_profile_executor.rs"]
mod inference_current_profile_executor;

use inference_binding::QuotaStateBinding;
use inference_contract::{
    InferencePolicy, InferencePurpose, InferenceRequest, InferenceRequirements, InformationClass,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_credential::InferenceCredentialLease;
use inference_current_profile_executor::{
    CurrentProfileBindingError, CurrentProfileCredentialExecutor,
};
use inference_execution_envelope::InferenceGenerationControls;
use inference_executor::{InferenceExecutionFailure, InferenceTickSource};
use inference_permit::InferencePermitIssuer;
use inference_profile_execution::{
    InferenceProfileExecutionError, ProviderProfileResolveError, ProviderProfileResolver,
    admit_and_bind_profile_execution,
};
use inference_provider_registry::{
    ProviderClaimEvidence, ProviderClaimSourceKind, ProviderDataPolicyClaim,
    ProviderEndpointDescriptor, ProviderModelCapabilities, ProviderModelLifecycle,
    ProviderModelProfile, ProviderProfileKey, ProviderQualificationPolicy, ProviderRegistry,
    ProviderRequestCostBound, QualifiedProviderCandidate, SourcedProviderClaim,
    digest_provider_snapshot,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const PROVIDER: &str = "example-provider";
const MODEL: &str = "reasoner-v1";
const DEPLOYMENT: &str = "deployment-a";

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
struct SharedRegistry(Arc<Mutex<ProviderRegistry>>);

impl SharedRegistry {
    fn install(&self, profile: ProviderModelProfile) {
        self.0.lock().unwrap().install(profile).unwrap();
    }
}

impl ProviderProfileResolver for SharedRegistry {
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

fn key() -> ProviderProfileKey {
    ProviderProfileKey {
        provider_id: PROVIDER.to_string(),
        deployment_id: DEPLOYMENT.to_string(),
        model_id: MODEL.to_string(),
    }
}

fn evidence(
    kind: ProviderClaimSourceKind,
    label: &str,
    valid_until_tick: u64,
) -> ProviderClaimEvidence {
    ProviderClaimEvidence::new(
        kind,
        format!("source:{label}"),
        digest_provider_snapshot(label.as_bytes()),
        90,
        valid_until_tick,
    )
    .unwrap()
}

fn profile(
    epoch: u64,
    base_url: &str,
    deployment_id: &str,
    account_scope: &str,
    salt: &str,
    valid_until_tick: u64,
) -> ProviderModelProfile {
    ProviderModelProfile::new(
        PROVIDER,
        deployment_id,
        MODEL,
        epoch,
        SourcedProviderClaim::new(
            ProviderEndpointDescriptor::openai_compatible(base_url).unwrap(),
            evidence(
                ProviderClaimSourceKind::FirstPartyModelCatalog,
                &format!("endpoint:{salt}"),
                valid_until_tick,
            ),
        ),
        SourcedProviderClaim::new(
            ProviderModelLifecycle::Production,
            evidence(
                ProviderClaimSourceKind::FirstPartyModelCatalog,
                &format!("lifecycle:{salt}"),
                valid_until_tick,
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
            evidence(
                ProviderClaimSourceKind::FirstPartyModelCatalog,
                &format!("capabilities:{salt}"),
                valid_until_tick,
            ),
        ),
        SourcedProviderClaim::new(
            ProviderDataPolicyClaim {
                training_policy: ProviderTrainingPolicy::Never,
                retention_policy: ProviderRetentionPolicy::ZeroRetention,
                routing_policy: ProviderRoutingPolicy::DirectOnly,
            },
            evidence(
                ProviderClaimSourceKind::FirstPartyPolicyDocument,
                &format!("policy:{salt}"),
                valid_until_tick,
            ),
        ),
        SourcedProviderClaim::new(
            ProviderRequestCostBound::new(account_scope, 0).unwrap(),
            evidence(
                ProviderClaimSourceKind::FirstPartyAccountState,
                &format!("cost:{salt}"),
                valid_until_tick,
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
    InferenceRequest::new("explain the profile invariant", requirements)
}

fn quota() -> QuotaStateBinding {
    QuotaStateBinding::new("quota:example-provider:deployment-a", 1)
        .unwrap()
        .with_remaining_requests(Some(10))
        .with_remaining_tokens(Some(10_000))
}

fn make_executor(
    registry: SharedRegistry,
) -> (
    CurrentProfileCredentialExecutor<SharedTick, SharedRegistry>,
    Arc<SharedTick>,
) {
    let clock = Arc::new(SharedTick::new(100));
    let lease = InferenceCredentialLease::anonymous("credential:example-provider:a", 1).unwrap();
    let executor = CurrentProfileCredentialExecutor::from_current_profile(
        registry,
        key(),
        ProviderQualificationPolicy::default(),
        lease,
        Duration::from_secs(2),
        1,
        clock.clone(),
    )
    .unwrap();
    (executor, clock)
}

fn prepare_current(
    executor: &CurrentProfileCredentialExecutor<SharedTick, SharedRegistry>,
) -> inference_permit::PreparedInferenceExecution {
    let bound = executor
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 20, [91; 32]).unwrap();
    issuer.prepare_execution(permit, binding, 101).unwrap()
}

#[test]
fn unchanged_profile_requalification_reproduces_same_v3_binding() {
    let base_url = "https://example.test/v1";
    let registry = SharedRegistry::default();
    registry.install(profile(1, base_url, DEPLOYMENT, "account-a", "same", 500));
    let (executor, clock) = make_executor(registry);

    let first = executor
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .unwrap();
    clock.set(102);
    let second = executor
        .bind_current(
            &InferencePolicy::free_private(),
            &request(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .unwrap();

    assert_eq!(first.binding(), second.binding());
}

#[tokio::test]
async fn newer_profile_supersedes_still_fresh_prepared_binding_before_network() {
    let dead_url = "http://127.0.0.1:9/v1";
    let registry = SharedRegistry::default();
    registry.install(profile(1, dead_url, DEPLOYMENT, "account-a", "epoch-1", 500));
    let (executor, clock) = make_executor(registry.clone());
    let prepared = prepare_current(&executor);

    registry.install(profile(2, dead_url, DEPLOYMENT, "account-a", "epoch-2", 500));
    clock.set(102);

    let outcome = executor
        .execute_current(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::BoundStateChanged));
    assert!(outcome.response_text.is_none());
    assert!(outcome.wire_observation.is_none());
}

#[tokio::test]
async fn profile_expiry_consumes_prepared_attempt_locally_without_network() {
    let dead_url = "http://127.0.0.1:9/v1";
    let registry = SharedRegistry::default();
    registry.install(profile(1, dead_url, DEPLOYMENT, "account-a", "short", 102));
    let (executor, clock) = make_executor(registry);
    let prepared = prepare_current(&executor);
    clock.set(102);

    let outcome = executor
        .execute_current(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        outcome.failure,
        Some(InferenceExecutionFailure::ProviderProfileRejected)
    );
    assert!(outcome.response_text.is_none());
    assert!(outcome.wire_observation.is_none());
    assert!(outcome.receipt.response_digest().is_none());
}

#[test]
fn deployment_and_account_scope_are_part_of_v3_provider_binding() {
    let base_url = "https://example.test/v1";
    let first = profile(1, base_url, "deployment-a", "account-a", "a", 500)
        .qualify(100, ProviderQualificationPolicy::default())
        .unwrap();
    let second = profile(1, base_url, "deployment-b", "account-b", "b", 500)
        .qualify(100, ProviderQualificationPolicy::default())
        .unwrap();

    let credential = inference_binding::CredentialStateBinding::new("credential", 1).unwrap();
    let endpoint = inference_execution_envelope::EndpointStateBinding::openai_compatible(
        PROVIDER,
        "https://example.test/v1/",
        MODEL,
        2_000,
        inference_execution_envelope::EndpointCredentialMode::None,
        1,
    )
    .unwrap();

    let first_bound = admit_and_bind_profile_execution(
        &InferencePolicy::free_private(),
        &request(),
        &first,
        &credential,
        &quota(),
        &endpoint,
        InferenceGenerationControls::default(),
        100,
    )
    .unwrap();
    let second_bound = admit_and_bind_profile_execution(
        &InferencePolicy::free_private(),
        &request(),
        &second,
        &credential,
        &quota(),
        &endpoint,
        InferenceGenerationControls::default(),
        100,
    )
    .unwrap();

    assert_ne!(
        first_bound.binding().provider_state_digest,
        second_bound.binding().provider_state_digest
    );
}

#[tokio::test]
async fn endpoint_drift_after_preparation_is_rejected_before_network() {
    let initial_url = "http://127.0.0.1:9/v1";
    let changed_url = "http://127.0.0.1:8/v1";
    let registry = SharedRegistry::default();
    registry.install(profile(1, initial_url, DEPLOYMENT, "account-a", "initial", 500));
    let (executor, clock) = make_executor(registry.clone());
    let prepared = prepare_current(&executor);

    registry.install(profile(2, changed_url, DEPLOYMENT, "account-a", "changed", 500));
    clock.set(102);

    let outcome = executor
        .execute_current(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        outcome.failure,
        Some(InferenceExecutionFailure::ProviderProfileRejected)
    );
    assert!(outcome.response_text.is_none());
    assert!(outcome.wire_observation.is_none());
}

#[tokio::test]
async fn unchanged_current_profile_executes_real_http_and_receipts() {
    let (base_url, captured) = spawn_one_shot_openai_server("profile-qualified response").await;
    let registry = SharedRegistry::default();
    registry.install(profile(1, &base_url, DEPLOYMENT, "account-a", "live", 500));
    let (executor, clock) = make_executor(registry);
    let prepared = prepare_current(&executor);
    clock.set(102);

    let outcome = executor
        .execute_current(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.failure, None);
    assert_eq!(
        outcome.response_text.as_deref(),
        Some("profile-qualified response")
    );
    assert!(outcome.receipt.response_digest().is_some());

    let request_text = captured.await.unwrap();
    assert!(request_text.contains("explain the profile invariant"));
    assert!(request_text.contains("\"model\":\"reasoner-v1\""));
}

#[test]
fn caller_cannot_construct_safe_executor_with_independent_endpoint_tuple() {
    let registry = SharedRegistry::default();
    registry.install(profile(
        1,
        "https://registry.example/v1",
        DEPLOYMENT,
        "account-a",
        "registry-owned",
        500,
    ));
    let (executor, _) = make_executor(registry);

    assert_eq!(executor.inner().endpoint_binding().provider_id(), PROVIDER);
    assert_eq!(executor.inner().endpoint_binding().wire_model(), MODEL);
    assert_eq!(
        executor.inner().endpoint_binding().base_url(),
        "https://registry.example/v1/"
    );
}

async fn spawn_one_shot_openai_server(
    response_text: &'static str,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let response_body = format!(
        "{{\"id\":\"resp-profile\",\"model\":\"reasoner-v1\",\"choices\":[{{\"finish_reason\":\"stop\",\"message\":{{\"content\":\"{}\"}}}}]}}",
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
