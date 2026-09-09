#[path = "../src/language/openai_compatible_transport.rs"]
mod openai_compatible_transport;

use openai_compatible_transport::{
    OpenAiCompatibleConfig, OpenAiCompatibleTransport, TransportCredential, TransportError,
    TransportGenerationRequest,
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn request(prompt: &str) -> TransportGenerationRequest {
    TransportGenerationRequest {
        prompt: prompt.to_string(),
        system_prompt: Some("system".to_string()),
        temperature: 0.7,
        max_tokens: 128,
    }
}

#[tokio::test]
async fn success_captures_provider_observations_without_prompt_echo() {
    let response_body = r#"{
        "id":"chatcmpl-observed-1",
        "model":"reasoner-v2",
        "system_fingerprint":"fp_abc",
        "choices":[{"finish_reason":"stop","message":{"content":"qualified response"}}],
        "usage":{"prompt_tokens":25,"completion_tokens":10,"total_tokens":35}
    }"#;
    let headers = [
        ("x-request-id", "req_header_1"),
        ("x-ratelimit-limit-requests", "1000"),
        ("x-ratelimit-remaining-requests", "999"),
        ("x-ratelimit-reset-requests", "2m59.56s"),
        ("x-ratelimit-limit-tokens", "18000"),
        ("x-ratelimit-remaining-tokens", "17997"),
        ("x-ratelimit-reset-tokens", "7.66s"),
    ];
    let base_url = spawn_one_shot_server(200, &headers, response_body).await;
    let transport = transport(&base_url);
    let secret_prompt = "private prompt that must not enter metadata";

    let observed = transport.generate_observed(&request(secret_prompt)).await.unwrap();
    assert_eq!(observed.response.text, "qualified response");
    assert_eq!(
        observed.observation.response_id.as_deref(),
        Some("chatcmpl-observed-1")
    );
    assert_eq!(
        observed.observation.provider_model.as_deref(),
        Some("reasoner-v2")
    );
    assert_eq!(
        observed.observation.system_fingerprint.as_deref(),
        Some("fp_abc")
    );
    assert_eq!(observed.observation.finish_reason.as_deref(), Some("stop"));
    assert_eq!(
        observed.observation.request_id_header.as_deref(),
        Some("req_header_1")
    );
    let usage = observed.observation.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(25));
    assert_eq!(usage.completion_tokens, Some(10));
    assert_eq!(usage.total_tokens, Some(35));
    assert_eq!(observed.observation.rate_limits.request_limit, Some(1000));
    assert_eq!(
        observed.observation.rate_limits.request_remaining,
        Some(999)
    );
    assert_eq!(
        observed.observation.rate_limits.request_reset_after_millis,
        Some(179_560)
    );
    assert_eq!(
        observed.observation.rate_limits.token_reset_after_millis,
        Some(7_660)
    );
    assert!(!observed.observation.metadata_conflict);
    assert!(!observed.observation.metadata_rejected);

    let rendered = format!("{:?}", observed.observation);
    assert!(!rendered.contains(secret_prompt));
    assert!(!rendered.contains("qualified response"));
}

#[tokio::test]
async fn rate_limited_failure_keeps_headers_but_discards_remote_body() {
    let echoed_secret = "provider echoed PRIVATE PROMPT in error body";
    let headers = [
        ("retry-after", "2"),
        ("x-ratelimit-remaining-requests", "0"),
        ("x-ratelimit-remaining-tokens", "17"),
    ];
    let base_url = spawn_one_shot_server(429, &headers, echoed_secret).await;
    let transport = transport(&base_url);

    let failure = transport
        .generate_observed(&request("PRIVATE PROMPT"))
        .await
        .unwrap_err();
    assert!(matches!(failure.error, TransportError::HttpStatus(429)));
    assert_eq!(
        failure.observation.rate_limits.retry_after_millis,
        Some(2_000)
    );
    assert_eq!(
        failure.observation.rate_limits.request_remaining,
        Some(0)
    );
    assert_eq!(
        failure.observation.rate_limits.token_remaining,
        Some(17)
    );

    let rendered = format!("{failure:?}");
    assert!(!rendered.contains(echoed_secret));
    assert!(!rendered.contains("PRIVATE PROMPT"));
}

#[tokio::test]
async fn legacy_api_remains_response_only_compatible() {
    let response_body = r#"{"choices":[{"message":{"content":"legacy response"}}]}"#;
    let base_url = spawn_one_shot_server(200, &[], response_body).await;
    let transport = transport(&base_url);

    let response = transport.generate(&request("hello")).await.unwrap();
    assert_eq!(response.text, "legacy response");
}

fn transport(base_url: &str) -> OpenAiCompatibleTransport {
    let config = OpenAiCompatibleConfig::new(
        "example-provider",
        base_url,
        "reasoner-v1",
        TransportCredential::None,
    )
    .unwrap()
    .with_timeout(Duration::from_secs(5))
    .unwrap();
    OpenAiCompatibleTransport::new(config).unwrap()
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
