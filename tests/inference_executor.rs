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

use inference_binding::{CredentialStateBinding, QuotaStateBinding};
use inference_contract::{
    ExecutionLocation, InferenceCandidate, InferencePolicy, InferencePurpose, InferenceRequest,
    InferenceRequirements, InformationClass, ModelIdentity, ProviderRetentionPolicy,
    ProviderRoutingPolicy, ProviderTrainingPolicy,
};
use inference_execution_envelope::{
    EndpointCredentialMode, EndpointStateBinding, InferenceGenerationControls,
    admit_and_bind_execution, digest_execution_provider_state,
};
use inference_executor::{
    InferenceExecutionFailure, InferenceTickSource, OpenAiInferenceExecutor,
};
use inference_permit::InferencePermitIssuer;
use openai_compatible_transport::TransportCredential;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Debug, Clone, Copy)]
struct FixedTick(u64);

impl InferenceTickSource for FixedTick {
    fn now_tick(&self) -> u64 {
        self.0
    }
}

fn request() -> InferenceRequest {
    let mut requirements = InferenceRequirements::minimal(
        InferencePurpose::GeneralReasoning,
        InformationClass::RemoteSafeDerived,
    );
    requirements.estimated_input_tokens = 64;
    requirements.max_output_tokens = 256;
    InferenceRequest::new("explain the invariant", requirements)
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

fn quota(epoch: u64) -> QuotaStateBinding {
    QuotaStateBinding::new("quota:example-provider:project-a", epoch)
        .unwrap()
        .with_remaining_requests(Some(50))
        .with_remaining_tokens(Some(100_000))
}

fn endpoint(url: &str, timeout_millis: u64) -> EndpointStateBinding {
    EndpointStateBinding::openai_compatible(
        "example-provider",
        url,
        "reasoner-v1",
        timeout_millis,
        EndpointCredentialMode::None,
        1,
    )
    .unwrap()
}

#[test]
fn generation_controls_are_part_of_request_binding() {
    let policy = InferencePolicy::free_private();
    let a = admit_and_bind_execution(
        &policy,
        &request(),
        &candidate(),
        &credential(1),
        &quota(1),
        &endpoint("https://example.test/v1/", 5_000),
        InferenceGenerationControls::new(700, false).unwrap(),
    )
    .unwrap();
    let b = admit_and_bind_execution(
        &policy,
        &request(),
        &candidate(),
        &credential(1),
        &quota(1),
        &endpoint("https://example.test/v1/", 5_000),
        InferenceGenerationControls::new(701, false).unwrap(),
    )
    .unwrap();
    assert_ne!(a.binding().request_digest, b.binding().request_digest);
}

#[test]
fn endpoint_timeout_and_url_are_part_of_provider_binding() {
    let c = inference_binding::digest_candidate(&candidate()).unwrap();
    let first = digest_execution_provider_state(c, &endpoint("https://example.test/v1/", 5_000)).unwrap();
    let second = digest_execution_provider_state(c, &endpoint("https://example.test/v1/", 6_000)).unwrap();
    let third = digest_execution_provider_state(c, &endpoint("https://other.test/v1/", 5_000)).unwrap();
    assert_ne!(first, second);
    assert_ne!(first, third);
}

#[test]
fn endpoint_model_must_match_provider_attestation() {
    let wrong = EndpointStateBinding::openai_compatible(
        "example-provider",
        "https://example.test/v1/",
        "other-model",
        5_000,
        EndpointCredentialMode::None,
        1,
    )
    .unwrap();
    let err = admit_and_bind_execution(
        &InferencePolicy::free_private(),
        &request(),
        &candidate(),
        &credential(1),
        &quota(1),
        &wrong,
        InferenceGenerationControls::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("wire model"));
}

#[tokio::test]
async fn executor_rechecks_binding_before_network_io() {
    let executor = OpenAiInferenceExecutor::new(
        "example-provider",
        "http://127.0.0.1:9/v1",
        "reasoner-v1",
        TransportCredential::None,
        Duration::from_secs(1),
        1,
        FixedTick(102),
    )
    .unwrap();
    let controls = InferenceGenerationControls::default();
    let bound = admit_and_bind_execution(
        &InferencePolicy::free_private(),
        &request(),
        &candidate(),
        &credential(1),
        &quota(1),
        executor.endpoint_binding(),
        controls,
    )
    .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 10, [50; 32]).unwrap();
    let prepared = issuer.prepare_execution(permit, binding, 101).unwrap();

    // Change the credential epoch after preparation. The executor must fail before
    // trying the deliberately dead TCP endpoint.
    let outcome = executor
        .execute(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &candidate(),
            &credential(2),
            &quota(1),
            controls,
        )
        .await
        .unwrap();
    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::BoundStateChanged));
    assert!(outcome.response_text.is_none());
    assert!(outcome.receipt.response_digest().is_none());
}

#[tokio::test]
async fn successful_execution_is_derived_from_bound_semantics_and_receipted() {
    let (base_url, captured) = spawn_one_shot_openai_server("qualified response").await;
    let executor = OpenAiInferenceExecutor::new(
        "example-provider",
        &base_url,
        "reasoner-v1",
        TransportCredential::None,
        Duration::from_secs(5),
        7,
        FixedTick(102),
    )
    .unwrap();
    let controls = InferenceGenerationControls::new(700, false).unwrap();
    let policy = InferencePolicy::free_private();
    let req = request();
    let cand = candidate();
    let cred = credential(1);
    let quota = quota(1);
    let bound = admit_and_bind_execution(
        &policy,
        &req,
        &cand,
        &cred,
        &quota,
        executor.endpoint_binding(),
        controls,
    )
    .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 10, [51; 32]).unwrap();
    let prepared = issuer.prepare_execution(permit, binding, 101).unwrap();

    let outcome = executor
        .execute(prepared, &policy, &req, &cand, &cred, &quota, controls)
        .await
        .unwrap();
    assert_eq!(outcome.failure, None);
    assert_eq!(outcome.response_text.as_deref(), Some("qualified response"));
    assert!(outcome.receipt.response_digest().is_some());

    let request_text = captured.await.unwrap();
    assert!(request_text.contains("explain the invariant"));
    assert!(request_text.contains("\"model\":\"reasoner-v1\""));
    assert!(request_text.contains("\"temperature\":0.7"));
    assert!(request_text.contains("\"max_tokens\":256"));
}

async fn spawn_one_shot_openai_server(response_text: &'static str) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let response_body = format!(
        "{{\"choices\":[{{\"message\":{{\"content\":\"{}\"}}}}]}}",
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
