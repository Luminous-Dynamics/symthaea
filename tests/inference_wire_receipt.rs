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
use inference_executor::{InferenceExecutionFailure, InferenceTickSource, OpenAiInferenceExecutor};
use inference_permit::InferencePermitIssuer;
use inference_receipt::{
    InferenceFailureClass, InferenceRateLimitEvidence, InferenceReceipt,
    InferenceTokenUsageEvidence, InferenceWireEvidence,
};
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

fn prepared(
    executor: &OpenAiInferenceExecutor<FixedTick>,
    nonce: u8,
) -> inference_permit::PreparedInferenceExecution {
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
    let permit = issuer.issue(binding, 100, 10, [nonce; 32]).unwrap();
    issuer.prepare_execution(permit, binding, 101).unwrap()
}

#[tokio::test]
async fn success_receipt_digests_provider_ids_and_redacts_outcome_debug() {
    let response_text = "sensitive generated answer";
    let response_body = format!(
        "{{\"id\":\"chatcmpl-provider-secret-1\",\"model\":\"reasoner-v1\",\"system_fingerprint\":\"fp_provider_a\",\"choices\":[{{\"finish_reason\":\"stop\",\"message\":{{\"content\":\"{response_text}\"}}}}],\"usage\":{{\"prompt_tokens\":20,\"completion_tokens\":8,\"total_tokens\":28}}}}"
    );
    let headers = [
        ("x-request-id", "provider-request-secret-1"),
        ("x-ratelimit-limit-requests", "1000"),
        ("x-ratelimit-remaining-requests", "999"),
        ("x-ratelimit-reset-requests", "2m59.56s"),
        ("x-ratelimit-limit-tokens", "18000"),
        ("x-ratelimit-remaining-tokens", "17972"),
        ("x-ratelimit-reset-tokens", "7.66s"),
    ];
    let base_url = spawn_one_shot_server(200, &headers, &response_body).await;
    let executor = OpenAiInferenceExecutor::new(
        "example-provider",
        &base_url,
        "reasoner-v1",
        TransportCredential::None,
        Duration::from_secs(5),
        1,
        FixedTick(102),
    )
    .unwrap();

    let outcome = executor
        .execute(
            prepared(&executor, 61),
            &InferencePolicy::free_private(),
            &request(),
            &candidate(),
            &credential(1),
            &quota(1),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.response_text.as_deref(), Some(response_text));
    assert_eq!(outcome.failure, None);
    let immediate = outcome.wire_observation.as_ref().unwrap();
    assert_eq!(
        immediate.response_id.as_deref(),
        Some("chatcmpl-provider-secret-1")
    );
    assert_eq!(
        immediate.request_id_header.as_deref(),
        Some("provider-request-secret-1")
    );

    let evidence = outcome.receipt.wire_evidence().unwrap();
    assert!(evidence.provider_response_id_digest().is_some());
    assert!(evidence.provider_request_id_digest().is_some());
    assert_eq!(evidence.provider_declared_model(), Some("reasoner-v1"));
    assert_eq!(evidence.system_fingerprint(), Some("fp_provider_a"));
    assert_eq!(evidence.finish_reason(), Some("stop"));
    assert_eq!(
        evidence.usage(),
        Some(InferenceTokenUsageEvidence {
            prompt_tokens: Some(20),
            completion_tokens: Some(8),
            total_tokens: Some(28),
        })
    );
    assert_eq!(evidence.rate_limits().request_remaining, Some(999));
    assert_eq!(
        evidence.rate_limits().request_reset_after_millis,
        Some(179_560)
    );
    assert_eq!(
        evidence.rate_limits().token_reset_after_millis,
        Some(7_660)
    );

    let receipt_debug = format!("{:?}", outcome.receipt);
    assert!(!receipt_debug.contains("chatcmpl-provider-secret-1"));
    assert!(!receipt_debug.contains("provider-request-secret-1"));
    assert!(!receipt_debug.contains(response_text));

    let outcome_debug = format!("{outcome:?}");
    assert!(!outcome_debug.contains("chatcmpl-provider-secret-1"));
    assert!(!outcome_debug.contains("provider-request-secret-1"));
    assert!(!outcome_debug.contains(response_text));
    assert!(outcome_debug.contains("response_text_present"));
}

#[tokio::test]
async fn rate_limited_failure_receipt_keeps_quota_evidence_without_error_body() {
    let echoed_secret = "PRIVATE-PROMPT-ECHO-SHOULD-NOT-SURVIVE";
    let headers = [
        ("x-request-id", "request-429-secret"),
        ("retry-after", "3"),
        ("x-ratelimit-remaining-requests", "0"),
        ("x-ratelimit-remaining-tokens", "44"),
    ];
    let base_url = spawn_one_shot_server(429, &headers, echoed_secret).await;
    let executor = OpenAiInferenceExecutor::new(
        "example-provider",
        &base_url,
        "reasoner-v1",
        TransportCredential::None,
        Duration::from_secs(5),
        1,
        FixedTick(102),
    )
    .unwrap();

    let outcome = executor
        .execute(
            prepared(&executor, 62),
            &InferencePolicy::free_private(),
            &request(),
            &candidate(),
            &credential(1),
            &quota(1),
            InferenceGenerationControls::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.failure, Some(InferenceExecutionFailure::ProviderHttp));
    assert!(outcome.response_text.is_none());
    assert!(outcome.receipt.response_digest().is_none());
    let evidence = outcome.receipt.wire_evidence().unwrap();
    assert!(evidence.provider_request_id_digest().is_some());
    assert_eq!(evidence.rate_limits().retry_after_millis, Some(3_000));
    assert_eq!(evidence.rate_limits().request_remaining, Some(0));
    assert_eq!(evidence.rate_limits().token_remaining, Some(44));

    let debug = format!("{outcome:?}");
    assert!(!debug.contains(echoed_secret));
    assert!(!debug.contains("request-429-secret"));
}

#[test]
fn different_provider_ids_produce_different_receipt_digests() {
    let rate_limits = InferenceRateLimitEvidence::default();
    let first = InferenceWireEvidence::new(
        Some("provider-response-a"),
        Some("provider-request-a"),
        Some("reasoner-v1"),
        None,
        Some("stop"),
        None,
        rate_limits,
        1,
        false,
        false,
    )
    .unwrap();
    let second = InferenceWireEvidence::new(
        Some("provider-response-b"),
        Some("provider-request-b"),
        Some("reasoner-v1"),
        None,
        Some("stop"),
        None,
        rate_limits,
        1,
        false,
        false,
    )
    .unwrap();

    assert_ne!(
        first.provider_response_id_digest(),
        second.provider_response_id_digest()
    );
    assert_ne!(
        first.provider_request_id_digest(),
        second.provider_request_id_digest()
    );
}

#[test]
fn legacy_receipt_constructor_keeps_wire_evidence_absent() {
    let policy = InferencePolicy::free_private();
    let req = request();
    let cand = candidate();
    let cred = credential(1);
    let quota = quota(1);

    // Build a standalone semantic permit to prove existing IF-3 constructors stay valid.
    let bound = inference_binding::admit_and_bind(&policy, &req, &cand, &cred, &quota).unwrap();
    let binding = *bound.binding();
    let mut issuer = InferencePermitIssuer::new();
    let permit = issuer.issue(binding, 100, 10, [63; 32]).unwrap();
    let prepared = issuer.prepare_execution(permit, binding, 101).unwrap();
    let receipt = InferenceReceipt::failure(prepared, InferenceFailureClass::Cancelled, 102).unwrap();
    assert!(receipt.wire_evidence().is_none());
}

async fn spawn_one_shot_server(
    status: u16,
    headers: &[(&str, &str)],
    response_body: &str,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let response_body = response_body.to_string();
    let headers: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();

    tokio::spawn(async move {
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
                let header_text = String::from_utf8_lossy(&request[..header_end]);
                let content_length = header_text
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

        let reason = if status == 200 {
            "OK"
        } else if status == 429 {
            "Too Many Requests"
        } else {
            "Status"
        };
        let mut response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            response_body.len()
        );
        for (name, value) in headers {
            response.push_str(&format!("{name}: {value}\r\n"));
        }
        response.push_str("\r\n");
        response.push_str(&response_body);
        stream.write_all(response.as_bytes()).await.unwrap();
    });

    format!("http://{address}/v1")
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}
