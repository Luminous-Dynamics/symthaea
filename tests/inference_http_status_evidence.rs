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
use inference_execution_envelope::{InferenceGenerationControls, admit_and_bind_execution};
use inference_executor::{InferenceExecutionFailure, InferenceExecutionOutcome, InferenceTickSource, OpenAiInferenceExecutor};
use inference_permit::InferencePermitIssuer;
use openai_compatible_transport::TransportCredential;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const PROVIDER: &str = "status-provider";
const MODEL: &str = "reasoner-v1";
const SECRET_ECHO: &str = "PRIVATE_PROMPT_ECHO_MUST_NOT_SURVIVE";

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
    requirements.estimated_input_tokens = 32;
    requirements.max_output_tokens = 64;
    InferenceRequest::new("status evidence request", requirements)
}

fn candidate() -> InferenceCandidate {
    InferenceCandidate {
        provider_id: PROVIDER.to_owned(),
        model: ModelIdentity::ProviderAttested {
            provider: PROVIDER.to_owned(),
            declared_model: MODEL.to_owned(),
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

fn credential() -> CredentialStateBinding {
    CredentialStateBinding::new("credential:status-provider", 1).unwrap()
}

fn quota() -> QuotaStateBinding {
    QuotaStateBinding::new("quota:status-provider", 1)
        .unwrap()
        .with_remaining_requests(Some(10))
        .with_remaining_tokens(Some(10_000))
}

fn prepare(
    executor: &OpenAiInferenceExecutor<FixedTick>,
) -> inference_permit::PreparedInferenceExecution {
    let policy = InferencePolicy::free_private();
    let request = request();
    let candidate = candidate();
    let credential = credential();
    let quota = quota();
    let controls = InferenceGenerationControls::default();
    let bound = admit_and_bind_execution(
        &policy,
        &request,
        &candidate,
        &credential,
        &quota,
        executor.endpoint_binding(),
        controls,
    )
    .unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 20, [141; 32]).unwrap();
    issuer.prepare_execution(permit, binding, 101).unwrap()
}

async fn execute_against_status(status: u16, extra_headers: &str) -> InferenceExecutionOutcome {
    let base_url = spawn_one_shot_status_server(status, extra_headers).await;
    let executor = OpenAiInferenceExecutor::new(
        PROVIDER,
        &base_url,
        MODEL,
        TransportCredential::None,
        Duration::from_secs(2),
        1,
        FixedTick(102),
    )
    .unwrap();
    let prepared = prepare(&executor);
    executor
        .execute(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &candidate(),
            &credential(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn failed_http_status_is_retained_without_remote_error_body() {
    let outcome = execute_against_status(
        429,
        "Retry-After: 2\r\nX-RateLimit-Remaining-Requests: 0\r\n",
    )
    .await;

    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::ProviderHttp));
    let evidence = outcome.receipt.wire_evidence().unwrap();
    assert_eq!(evidence.provider_http_status(), Some(429));
    assert_eq!(evidence.rate_limits().retry_after_millis, Some(2_000));
    assert_eq!(evidence.rate_limits().request_remaining, Some(0));

    let debug = format!("{outcome:?}");
    assert!(!debug.contains(SECRET_ECHO));
    assert!(!format!("{:?}", outcome.receipt).contains(SECRET_ECHO));
}

#[tokio::test]
async fn auth_and_server_statuses_remain_distinguishable_for_future_settlement() {
    for status in [401_u16, 403, 500] {
        let outcome = execute_against_status(status, "").await;
        assert_eq!(outcome.failure, Some(InferenceExecutionFailure::ProviderHttp));
        assert_eq!(
            outcome
                .receipt
                .wire_evidence()
                .unwrap()
                .provider_http_status(),
            Some(status)
        );
    }
}

#[tokio::test]
async fn non_http_transport_failure_does_not_invent_status() {
    let executor = OpenAiInferenceExecutor::new(
        PROVIDER,
        "http://127.0.0.1:9/v1",
        MODEL,
        TransportCredential::None,
        Duration::from_millis(100),
        1,
        FixedTick(102),
    )
    .unwrap();
    let prepared = prepare(&executor);
    let outcome = executor
        .execute(
            prepared,
            &InferencePolicy::free_private(),
            &request(),
            &candidate(),
            &credential(),
            &quota(),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::Transport));
    assert_eq!(
        outcome
            .receipt
            .wire_evidence()
            .unwrap()
            .provider_http_status(),
        None
    );
}

async fn spawn_one_shot_status_server(status: u16, extra_headers: &str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let extra_headers = extra_headers.to_owned();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request_bytes = Vec::new();
        let mut scratch = [0_u8; 4096];
        loop {
            let count = stream.read(&mut scratch).await.unwrap();
            if count == 0 {
                break;
            }
            request_bytes.extend_from_slice(&scratch[..count]);
            if request_bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }

        let body = format!("{{\"error\":\"{SECRET_ECHO}\"}}");
        let reason = match status {
            401 => "Unauthorized",
            403 => "Forbidden",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "Error",
        };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    format!("http://{address}/v1")
}
