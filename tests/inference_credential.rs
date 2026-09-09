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
#[path = "../src/language/inference_executor.rs"]
mod inference_executor;
#[path = "../src/language/inference_credential.rs"]
mod inference_credential;

use inference_binding::{CredentialStateBinding, QuotaStateBinding};
use inference_contract::{
    ExecutionLocation, InferenceCandidate, InferencePolicy, InferencePurpose, InferenceRequest,
    InferenceRequirements, InformationClass, ModelIdentity, ProviderRetentionPolicy,
    ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_credential::{
    CredentialBoundOpenAiExecutor, InferenceCredentialHandle, InferenceCredentialLease,
    InferenceCredentialMode, InferenceCredentialResolveError, InferenceCredentialResolver,
    resolve_credential,
};
use inference_execution_envelope::{InferenceGenerationControls, admit_and_bind_execution};
use inference_executor::{InferenceExecutionFailure, InferenceTickSource};
use inference_permit::InferencePermitIssuer;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Debug, Clone, Copy)]
struct FixedTick(u64);
impl InferenceTickSource for FixedTick {
    fn now_tick(&self) -> u64 { self.0 }
}

fn request() -> InferenceRequest {
    let mut requirements = InferenceRequirements::minimal(
        InferencePurpose::GeneralReasoning,
        InformationClass::RemoteSafeDerived,
    );
    requirements.estimated_input_tokens = 32;
    requirements.max_output_tokens = 64;
    InferenceRequest::new("credential-bound request", requirements)
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
        context_window_tokens: 4_096,
        supports_streaming: true,
        supports_tools: false,
        supports_structured_output: false,
        training_policy: ProviderTrainingPolicy::Never,
        retention_policy: ProviderRetentionPolicy::ZeroRetention,
        routing_policy: ProviderRoutingPolicy::DirectOnly,
        max_charge_microusd: Some(0),
    }
}

fn quota() -> QuotaStateBinding {
    QuotaStateBinding::new("quota:example-provider:test", 1)
        .unwrap()
        .with_remaining_requests(Some(100))
        .with_remaining_tokens(Some(10_000))
}

#[test]
fn lease_debug_redacts_bearer_secret() {
    let lease = InferenceCredentialLease::bearer("credential:example:primary", 7, "super-secret")
        .unwrap();
    let rendered = format!("{lease:?}");
    assert!(!rendered.contains("super-secret"));
    assert!(rendered.contains("credential:example:primary"));
    assert!(rendered.contains("Bearer"));
}

struct WrongResolver;
impl InferenceCredentialResolver for WrongResolver {
    fn resolve(
        &self,
        _handle: &InferenceCredentialHandle,
    ) -> Result<InferenceCredentialLease, InferenceCredentialResolveError> {
        Ok(InferenceCredentialLease::bearer("credential:wrong", 99, "dummy").unwrap())
    }
}

#[test]
fn resolver_cannot_substitute_another_identity_epoch_or_mode() {
    let handle = InferenceCredentialHandle::new(
        "credential:expected",
        7,
        InferenceCredentialMode::Bearer,
    )
    .unwrap();
    let error = resolve_credential(&WrongResolver, &handle).unwrap_err();
    assert_eq!(error, InferenceCredentialResolveError::BindingMismatch);
}

#[tokio::test]
async fn correctly_bound_lease_drives_wire_authorization_and_receipt() {
    let (base_url, captured) = spawn_one_shot_server("bound response").await;
    let executor = CredentialBoundOpenAiExecutor::from_lease(
        "example-provider",
        &base_url,
        "reasoner-v1",
        InferenceCredentialLease::bearer("credential:example:primary", 7, "dummy-token").unwrap(),
        Duration::from_secs(5),
        1,
        FixedTick(103),
    )
    .unwrap();

    let policy = InferencePolicy::free_private();
    let req = request();
    let cand = candidate();
    let quota = quota();
    let controls = InferenceGenerationControls::new(700, false).unwrap();
    let bound = admit_and_bind_execution(
        &policy,
        &req,
        &cand,
        executor.credential_binding(),
        &quota,
        executor.endpoint_binding(),
        controls,
    )
    .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 10, [70; 32]).unwrap();
    let prepared = issuer.prepare_execution(permit, binding, 101).unwrap();

    let outcome = executor
        .execute(prepared, &policy, &req, &cand, &quota, controls)
        .await
        .unwrap();
    assert_eq!(outcome.failure, None);
    assert_eq!(outcome.response_text.as_deref(), Some("bound response"));
    assert!(outcome.receipt.response_digest().is_some());

    let request_text = captured.await.unwrap().to_ascii_lowercase();
    assert!(request_text.contains("authorization: bearer dummy-token"));
}

#[tokio::test]
async fn separately_forged_credential_binding_is_rejected_before_network() {
    let executor = CredentialBoundOpenAiExecutor::from_lease(
        "example-provider",
        "http://127.0.0.1:9/v1",
        "reasoner-v1",
        InferenceCredentialLease::bearer("credential:real", 7, "dummy-token").unwrap(),
        Duration::from_secs(1),
        1,
        FixedTick(103),
    )
    .unwrap();

    let policy = InferencePolicy::free_private();
    let req = request();
    let cand = candidate();
    let quota = quota();
    let controls = InferenceGenerationControls::default();
    let fake = CredentialStateBinding::new("credential:fake", 7).unwrap();

    let bound = admit_and_bind_execution(
        &policy,
        &req,
        &cand,
        &fake,
        &quota,
        executor.endpoint_binding(),
        controls,
    )
    .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 10, [71; 32]).unwrap();
    let prepared = issuer.prepare_execution(permit, binding, 101).unwrap();

    let outcome = executor
        .execute(prepared, &policy, &req, &cand, &quota, controls)
        .await
        .unwrap();
    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::BoundStateChanged));
    assert!(outcome.response_text.is_none());
    assert!(outcome.wire_observation.is_none());
}

async fn spawn_one_shot_server(
    response_text: &'static str,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let response_body = format!(
        "{{\"id\":\"response-1\",\"model\":\"reasoner-v1\",\"choices\":[{{\"finish_reason\":\"stop\",\"message\":{{\"content\":\"{}\"}}}}]}}",
        response_text
    );
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut scratch = [0_u8; 4096];
        let mut expected_total = None;
        loop {
            let count = stream.read(&mut scratch).await.unwrap();
            if count == 0 { break; }
            request.extend_from_slice(&scratch[..count]);
            if expected_total.is_none()
                && let Some(header_end) = request.windows(4).position(|w| w == b"\r\n\r\n")
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
            if expected_total.is_some_and(|total| request.len() >= total) { break; }
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
