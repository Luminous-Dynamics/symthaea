#![allow(dead_code, unused_imports)]

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
#[path = "../src/language/inference_resource_guard.rs"]
mod inference_resource_guard;
#[path = "../src/language/inference_executor.rs"]
mod inference_executor;
#[path = "../src/language/inference_credential.rs"]
mod inference_credential;
#[path = "../src/language/inference_current_profile_executor.rs"]
mod inference_current_profile_executor;
#[path = "../src/language/inference_resource_execution.rs"]
mod inference_resource_execution;
#[path = "../src/language/inference_resource_lease.rs"]
mod inference_resource_lease;
#[path = "../src/language/inference_leased_executor.rs"]
mod inference_leased_executor;

use inference_binding::{CredentialStateBinding, QuotaStateBinding};
use inference_contract::{
    InferencePolicy, InferencePurpose, InferenceRequest, InferenceRequirements, InformationClass,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_credential::InferenceCredentialLease;
use inference_current_profile_executor::CurrentProfileCredentialExecutor;
use inference_execution_envelope::InferenceGenerationControls;
use inference_executor::{InferenceExecutionFailure, InferenceTickSource};
use inference_leased_executor::{
    InferenceLeasedExecutionError, InferenceLeasedExecutorError, InferenceResourceMillisSource,
    LeasedCurrentProfileExecutor, PreparedLeasedInferenceExecution,
};
use inference_permit::{InferencePermitError, InferencePermitIssuer};
use inference_profile_execution::{ProviderProfileResolveError, ProviderProfileResolver};
use inference_provider_registry::{
    ProviderClaimEvidence, ProviderClaimSourceKind, ProviderDataPolicyClaim,
    ProviderEndpointDescriptor, ProviderModelCapabilities, ProviderModelLifecycle,
    ProviderModelProfile, ProviderProfileKey, ProviderQualificationPolicy, ProviderRegistry,
    ProviderRequestCostBound, QualifiedProviderCandidate, SourcedProviderClaim,
    digest_provider_snapshot,
};
use inference_receipt::{InferenceCompletion, InferenceFailureClass};
use inference_resource_guard::{
    InferenceResourceAuthority, InferenceResourceGuard, InferenceResourceGuardConfig,
    InferenceResourceGuardError,
};
use inference_resource_lease::{InferenceExecutionResourceAuthority, InferenceResourceLeaseError};
use inference_resource_scope::{
    InferenceResourceScopeKey, InferenceResourceScopeRegistry, LocalScopeRegistrationEvidence,
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
struct SharedClock {
    tick: AtomicU64,
    millis: AtomicU64,
}

impl SharedClock {
    fn new(tick: u64, millis: u64) -> Self {
        Self {
            tick: AtomicU64::new(tick),
            millis: AtomicU64::new(millis),
        }
    }

    fn set_tick(&self, tick: u64) {
        self.tick.store(tick, Ordering::SeqCst);
    }

    fn set_millis(&self, millis: u64) {
        self.millis.store(millis, Ordering::SeqCst);
    }
}

impl InferenceTickSource for SharedClock {
    fn now_tick(&self) -> u64 {
        self.tick.load(Ordering::SeqCst)
    }
}

impl InferenceResourceMillisSource for SharedClock {
    fn now_millis(&self) -> u64 {
        self.millis.load(Ordering::SeqCst)
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

fn credential(epoch: u64) -> CredentialStateBinding {
    CredentialStateBinding::new(CREDENTIAL_ID, epoch).unwrap()
}

fn quota(epoch: u64) -> QuotaStateBinding {
    QuotaStateBinding::new(QUOTA_ID, epoch)
        .unwrap()
        .with_remaining_requests(Some(100))
        .with_remaining_tokens(Some(100_000))
}

fn request() -> InferenceRequest {
    let mut requirements = InferenceRequirements::minimal(
        InferencePurpose::GeneralReasoning,
        InformationClass::RemoteSafeDerived,
    );
    requirements.estimated_input_tokens = 64;
    requirements.max_output_tokens = 128;
    InferenceRequest::new("explain the execution resource lease invariant", requirements)
}

fn changed_request() -> InferenceRequest {
    let mut value = request();
    value.prompt = "different request after permit preparation".to_owned();
    value
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

fn profile(base_url: &str, salt: &str) -> ProviderModelProfile {
    ProviderModelProfile::new(
        PROVIDER,
        DEPLOYMENT,
        MODEL,
        1,
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

fn local_evidence(epoch: u64, label: &str) -> LocalScopeRegistrationEvidence {
    LocalScopeRegistrationEvidence::new(epoch, *blake3::hash(label.as_bytes()).as_bytes()).unwrap()
}

fn scope_registry() -> InferenceResourceScopeRegistry {
    let mut registry = InferenceResourceScopeRegistry::new([41; 32]).unwrap();
    registry
        .register_credential(&credential(1), scope(), local_evidence(1, "credential"))
        .unwrap();
    registry
        .register_quota(&quota(1), scope(), local_evidence(1, "quota"))
        .unwrap();
    registry
}

fn guard(
    request_budget: u64,
    token_budget: u64,
    failure_threshold: u32,
    instance_byte: u8,
) -> InferenceResourceGuard {
    let mut config = InferenceResourceGuardConfig::default();
    config.failure_threshold = failure_threshold;
    config.failure_base_backoff_millis = 500;
    config.default_rate_limit_backoff_millis = 500;
    config.max_backoff_millis = 5_000;
    InferenceResourceGuard::new(
        InferenceResourceAuthority::new(
            1,
            1,
            Some(request_budget),
            Some(token_budget),
        ),
        config,
        [instance_byte; 32],
    )
    .unwrap()
}

type TestExecutor = LeasedCurrentProfileExecutor<SharedClock, SharedProviderRegistry, SharedClock>;

fn make_executor(
    base_url: &str,
    request_budget: u64,
    token_budget: u64,
    failure_threshold: u32,
    authority_byte: u8,
) -> (TestExecutor, Arc<SharedClock>, QuotaStateBinding) {
    let providers = SharedProviderRegistry::default();
    providers.install(profile(base_url, &format!("authority-{authority_byte}")));
    let clock = Arc::new(SharedClock::new(100, 1_000));
    let current = CurrentProfileCredentialExecutor::from_current_profile(
        providers,
        key(),
        ProviderQualificationPolicy::default(),
        InferenceCredentialLease::anonymous(CREDENTIAL_ID, 1).unwrap(),
        Duration::from_secs(2),
        1,
        clock.clone(),
    )
    .unwrap();
    let authority = InferenceExecutionResourceAuthority::new(
        [authority_byte; 32],
        guard(
            request_budget,
            token_budget,
            failure_threshold,
            authority_byte.wrapping_add(70),
        ),
        scope_registry(),
    )
    .unwrap();
    (
        LeasedCurrentProfileExecutor::new(current, authority, clock.clone()),
        clock,
        quota(1),
    )
}

fn prepare(
    executor: &TestExecutor,
    quota: &QuotaStateBinding,
    nonce: u8,
) -> PreparedLeasedInferenceExecution {
    let mut issuer = InferencePermitIssuer::new();
    executor
        .prepare_leased_current(
            &mut issuer,
            &InferencePolicy::free_private(),
            &request(),
            quota,
            InferenceGenerationControls::default(),
            20,
            [nonce; 32],
        )
        .unwrap()
}

#[test]
fn permit_failure_retires_lease_without_refunding_reserved_capacity() {
    let (executor, _, quota) = make_executor("https://example.test/v1", 1, 1_000, 3, 10);
    let mut issuer = InferencePermitIssuer::new();
    let result = executor.prepare_leased_current(
        &mut issuer,
        &InferencePolicy::free_private(),
        &request(),
        &quota,
        InferenceGenerationControls::default(),
        0,
        [50; 32],
    );

    assert!(matches!(
        result,
        Err(InferenceLeasedExecutorError::Permit(InferencePermitError::ZeroTtl))
    ));
    assert_eq!(executor.active_lease_count().unwrap(), 0);
    assert_eq!(executor.effective_remaining_requests().unwrap(), Some(0));
    assert_eq!(executor.effective_remaining_tokens().unwrap(), Some(808));
}

#[test]
fn cancellation_retires_prepared_lease_without_refund() {
    let (executor, _, quota) = make_executor("https://example.test/v1", 2, 1_000, 3, 22);
    let capability = prepare(&executor, &quota, 65);
    assert_eq!(executor.active_lease_count().unwrap(), 1);

    let receipt = executor.cancel_prepared_leased_current(capability).unwrap();
    assert!(matches!(
        receipt.completion(),
        InferenceCompletion::Failure {
            class: InferenceFailureClass::Cancelled
        }
    ));
    assert_eq!(executor.active_lease_count().unwrap(), 0);
    assert_eq!(executor.effective_remaining_requests().unwrap(), Some(1));
    assert_eq!(executor.effective_remaining_tokens().unwrap(), Some(808));
}

#[test]
fn prepared_lease_excludes_second_caller_from_last_request() {
    let (executor, _, quota) = make_executor("https://example.test/v1", 1, 1_000, 3, 11);
    let _first = prepare(&executor, &quota, 51);

    assert_eq!(executor.active_lease_count().unwrap(), 1);
    assert_eq!(executor.effective_remaining_requests().unwrap(), Some(0));

    let mut issuer = InferencePermitIssuer::new();
    let second = executor.prepare_leased_current(
        &mut issuer,
        &InferencePolicy::free_private(),
        &request(),
        &quota,
        InferenceGenerationControls::default(),
        20,
        [52; 32],
    );
    assert!(matches!(
        second,
        Err(InferenceLeasedExecutorError::Lease(
            InferenceResourceLeaseError::Guard(InferenceResourceGuardError::RequestBudgetExhausted)
        ))
    ));
}

#[test]
fn active_prepared_capability_blocks_atomic_resource_state_replacement() {
    let (executor, _, quota) = make_executor("https://example.test/v1", 2, 1_000, 3, 12);
    let _capability = prepare(&executor, &quota, 53);

    let replacement = executor.replace_resource_state(guard(10, 10_000, 3, 99), scope_registry());
    assert!(matches!(
        replacement,
        Err(InferenceLeasedExecutorError::Lease(
            InferenceResourceLeaseError::ActiveLeasesPresent
        ))
    ));
}

#[tokio::test]
async fn successful_http_execution_settles_lease_without_refund() {
    let (base_url, captured) = spawn_server(200, vec![], success_body("leased success")).await;
    let (executor, clock, quota) = make_executor(&base_url, 2, 1_000, 3, 13);
    let capability = prepare(&executor, &quota, 54);
    clock.set_tick(102);
    clock.set_millis(1_100);

    let outcome = executor
        .execute_prepared_leased_current(
            capability,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.failure, None);
    assert_eq!(outcome.response_text.as_deref(), Some("leased success"));
    assert_eq!(executor.active_lease_count().unwrap(), 0);
    assert_eq!(executor.effective_remaining_requests().unwrap(), Some(1));
    assert_eq!(executor.effective_remaining_tokens().unwrap(), Some(808));
    assert!(captured.await.unwrap().contains("execution resource lease invariant"));
}

#[tokio::test]
async fn rate_limit_status_settles_into_cooldown_without_body_retention() {
    let headers = vec![
        ("Retry-After", "2"),
        ("X-RateLimit-Remaining-Requests", "0"),
    ];
    let (base_url, _) = spawn_server(
        429,
        headers,
        "{\"error\":\"PRIVATE_BODY_MUST_NOT_SURVIVE\"}".to_owned(),
    )
    .await;
    let (executor, clock, quota) = make_executor(&base_url, 3, 2_000, 3, 14);
    let capability = prepare(&executor, &quota, 55);
    clock.set_tick(102);
    clock.set_millis(1_100);

    let outcome = executor
        .execute_prepared_leased_current(
            capability,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::ProviderHttp));
    assert_eq!(
        outcome
            .receipt
            .wire_evidence()
            .and_then(|evidence| evidence.provider_http_status()),
        Some(429)
    );
    assert!(!format!("{outcome:?}").contains("PRIVATE_BODY_MUST_NOT_SURVIVE"));
    assert_eq!(executor.active_lease_count().unwrap(), 0);

    clock.set_millis(1_101);
    let mut issuer = InferencePermitIssuer::new();
    let retry = executor.prepare_leased_current(
        &mut issuer,
        &InferencePolicy::free_private(),
        &request(),
        &quota,
        InferenceGenerationControls::default(),
        20,
        [56; 32],
    );
    assert!(matches!(
        retry,
        Err(InferenceLeasedExecutorError::Lease(
            InferenceResourceLeaseError::Guard(InferenceResourceGuardError::CoolingDown)
        ))
    ));
}

#[tokio::test]
async fn 401_and_403_block_current_credential_epoch() {
    for (status, authority_byte, nonce) in [(401, 15, 57), (403, 16, 58)] {
        let (base_url, _) =
            spawn_server(status, vec![], "{\"error\":\"auth\"}".to_owned()).await;
        let (executor, clock, quota) = make_executor(&base_url, 3, 2_000, 3, authority_byte);
        let capability = prepare(&executor, &quota, nonce);
        clock.set_tick(102);

        let outcome = executor
            .execute_prepared_leased_current(
                capability,
                &InferencePolicy::free_private(),
                &request(),
                &quota,
                InferenceGenerationControls::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            outcome
                .receipt
                .wire_evidence()
                .and_then(|evidence| evidence.provider_http_status()),
            Some(status)
        );
        assert_eq!(executor.active_lease_count().unwrap(), 0);

        let mut issuer = InferencePermitIssuer::new();
        let retry = executor.prepare_leased_current(
            &mut issuer,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
            20,
            [nonce.wrapping_add(20); 32],
        );
        assert!(matches!(
            retry,
            Err(InferenceLeasedExecutorError::Lease(
                InferenceResourceLeaseError::Guard(InferenceResourceGuardError::CredentialBlocked)
            ))
        ));
    }
}

#[tokio::test]
async fn provider_500_settles_conservatively_and_opens_local_circuit() {
    let (base_url, _) = spawn_server(500, vec![], "{\"error\":\"server\"}".to_owned()).await;
    let (executor, clock, quota) = make_executor(&base_url, 3, 2_000, 1, 17);
    let capability = prepare(&executor, &quota, 59);
    clock.set_tick(102);
    clock.set_millis(2_000);

    let outcome = executor
        .execute_prepared_leased_current(
            capability,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::ProviderHttp));
    assert_eq!(executor.active_lease_count().unwrap(), 0);

    let mut issuer = InferencePermitIssuer::new();
    let retry = executor.prepare_leased_current(
        &mut issuer,
        &InferencePolicy::free_private(),
        &request(),
        &quota,
        InferenceGenerationControls::default(),
        20,
        [60; 32],
    );
    assert!(matches!(
        retry,
        Err(InferenceLeasedExecutorError::Lease(
            InferenceResourceLeaseError::Guard(InferenceResourceGuardError::CoolingDown)
        ))
    ));
}

#[tokio::test]
async fn changed_request_is_rejected_before_wire_and_reservation_is_not_refunded() {
    let (base_url, captured) = spawn_server(200, vec![], success_body("must not execute")).await;
    let (executor, clock, quota) = make_executor(&base_url, 2, 1_000, 3, 18);
    let capability = prepare(&executor, &quota, 61);
    clock.set_tick(102);

    let outcome = executor
        .execute_prepared_leased_current(
            capability,
            &InferencePolicy::free_private(),
            &changed_request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::BoundStateChanged));
    assert!(outcome.wire_observation.is_none());
    assert_eq!(executor.active_lease_count().unwrap(), 0);
    assert_eq!(executor.effective_remaining_requests().unwrap(), Some(1));
    assert_eq!(executor.effective_remaining_tokens().unwrap(), Some(808));
    assert!(!captured.is_finished());
    captured.abort();
}

#[test]
fn lease_generation_changes_v5_provider_state() {
    let (executor, _, quota) = make_executor("https://example.test/v1", 3, 2_000, 3, 19);
    let mut issuer = InferencePermitIssuer::new();
    let first = executor
        .prepare_leased_current(
            &mut issuer,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
            20,
            [62; 32],
        )
        .unwrap();
    let second = executor
        .prepare_leased_current(
            &mut issuer,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
            20,
            [63; 32],
        )
        .unwrap();
    assert_ne!(
        first.binding().provider_state_digest,
        second.binding().provider_state_digest
    );
    assert_ne!(first.permit_generation(), second.permit_generation());
}

#[tokio::test]
async fn wrong_authority_returns_capability_intact_for_origin_recovery() {
    let (base_url, captured) = spawn_server(200, vec![], success_body("origin recovered")).await;
    let (origin, origin_clock, quota) = make_executor(&base_url, 2, 2_000, 3, 20);
    let (other, _, _) = make_executor(&base_url, 2, 2_000, 3, 21);
    let capability = prepare(&origin, &quota, 64);

    let error = other
        .execute_prepared_leased_current(
            capability,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap_err();
    let capability = error
        .into_capability()
        .expect("wrong authority must return capability");
    assert_eq!(origin.active_lease_count().unwrap(), 1);
    assert_eq!(other.active_lease_count().unwrap(), 0);

    origin_clock.set_tick(102);
    let outcome = origin
        .execute_prepared_leased_current(
            capability,
            &InferencePolicy::free_private(),
            &request(),
            &quota,
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.response_text.as_deref(), Some("origin recovered"));
    assert_eq!(origin.active_lease_count().unwrap(), 0);
    assert!(captured.await.unwrap().contains("execution resource lease invariant"));
}

async fn spawn_server(
    status: u16,
    headers: Vec<(&'static str, &'static str)>,
    body: String,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
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
                let headers_text = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers_text
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

        let reason = match status {
            200 => "OK",
            401 => "Unauthorized",
            403 => "Forbidden",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "Status",
        };
        let mut response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            response.push_str(name);
            response.push_str(": ");
            response.push_str(value);
            response.push_str("\r\n");
        }
        response.push_str("\r\n");
        response.push_str(&body);
        stream.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8_lossy(&request).to_string()
    });
    (format!("http://{address}/v1"), handle)
}

fn success_body(text: &str) -> String {
    format!(
        "{{\"id\":\"resp-lease\",\"model\":\"{MODEL}\",\"choices\":[{{\"finish_reason\":\"stop\",\"message\":{{\"content\":\"{text}\"}}}}]}}"
    )
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}
