use std::path::PathBuf;

use rea_mcp_harness::ReaEvidenceMcpServer;
use rmcp::{
    ServiceExt,
    model::CallToolRequestParams,
};
use serde_json::{Value, json};

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("REA_UPSTREAM_EVIDENCE_FIXTURE_DIR").map(PathBuf::from)
}

async fn start_server() -> (
    rmcp::service::RunningService<rmcp::RoleClient, ()>,
    tokio::task::JoinHandle<()>,
) {
    let (server_transport, client_transport) = tokio::io::duplex(2 * 1024 * 1024);
    let task = tokio::spawn(async move {
        let service = ReaEvidenceMcpServer::new()
            .serve(server_transport)
            .await
            .expect("start in-memory MCP server");
        service.waiting().await.expect("MCP server shuts down cleanly");
    });
    let client = ()
        .serve(client_transport)
        .await
        .expect("connect in-memory MCP client");
    (client, task)
}

fn structured(result: &rmcp::model::CallToolResult) -> &Value {
    result
        .structured_content
        .as_ref()
        .expect("MCP tool returns structured JSON content")
}

#[tokio::test]
async fn server_exposes_only_two_read_only_verification_tools() {
    let (client, server_task) = start_server().await;

    let mut tools = client.list_all_tools().await.unwrap();
    tools.sort_by(|left, right| left.name.cmp(&right.name));
    assert_eq!(
        tools.iter().map(|tool| tool.name.as_ref()).collect::<Vec<_>>(),
        vec![
            "rea_verify_evidence_bundle_json",
            "rea_verify_evidence_json"
        ]
    );

    for tool in &tools {
        let annotations = tool.annotations.as_ref().expect("effect annotations");
        assert_eq!(annotations.read_only_hint, Some(true));
        assert_eq!(annotations.destructive_hint, Some(false));
        assert_eq!(annotations.idempotent_hint, Some(true));
        assert_eq!(annotations.open_world_hint, Some(false));
        let properties = tool.input_schema.get("properties").unwrap();
        assert_eq!(properties.as_object().unwrap().len(), 1);
        assert!(properties.get("path").is_none());
    }

    client.cancel().await.unwrap();
    server_task.await.unwrap();
}

#[tokio::test]
async fn evidence_tool_verifies_semantic_identity_and_redacts_payloads() {
    let raw = include_str!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    );
    let fixture: Value = serde_json::from_str(raw).unwrap();
    let (client, server_task) = start_server().await;
    let arguments = json!({ "evidence_json": raw });
    let response = client
        .call_tool(
            CallToolRequestParams::new("rea_verify_evidence_json")
                .with_arguments(arguments.as_object().unwrap().clone()),
        )
        .await
        .unwrap();

    assert_eq!(response.is_error, Some(false));
    let value = structured(&response);
    assert_eq!(value["status"], "verified_semantic_consistency");
    assert_eq!(value["evidence_id"], fixture["evidence_id"]);
    assert_eq!(value["claim_ceiling"]["producer_authenticity_verified"], false);
    assert_eq!(value["claim_ceiling"]["observation_truth_verified"], false);

    let serialized = value.to_string();
    assert!(!serialized.contains("/tmp/"));
    assert!(!serialized.contains("classes.dex"));
    assert!(!serialized.contains("raw_result"));
    assert!(!serialized.contains("normalized_result"));

    client.cancel().await.unwrap();
    server_task.await.unwrap();
}

#[tokio::test]
async fn rejected_evidence_returns_stable_category_without_echoing_input() {
    let (client, server_task) = start_server().await;
    let arguments = json!({
        "evidence_json": r#"{"same":1,"same":2}"#
    });
    let response = client
        .call_tool(
            CallToolRequestParams::new("rea_verify_evidence_json")
                .with_arguments(arguments.as_object().unwrap().clone()),
        )
        .await
        .unwrap();
    assert_eq!(response.is_error, Some(false));
    let value = structured(&response);
    assert_eq!(value["status"], "rejected");
    assert_eq!(value["error_code"], "invalid_json");
    assert!(!value.to_string().contains("same"));

    client.cancel().await.unwrap();
    server_task.await.unwrap();
}

#[tokio::test]
async fn bundle_tool_uses_the_upstream_generated_bundle_when_available() {
    let Some(directory) = fixture_dir() else {
        // Local focused harness runs may not have the upstream TS runtime corpus.
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
            .unwrap();
    let file = manifest["evidence_bundle"]["file"].as_str().unwrap();
    let expected = &manifest["evidence_bundle"];
    let bundle = std::fs::read_to_string(directory.join(file)).unwrap();

    let (client, server_task) = start_server().await;
    let arguments = json!({ "bundle_json": bundle });
    let response = client
        .call_tool(
            CallToolRequestParams::new("rea_verify_evidence_bundle_json")
                .with_arguments(arguments.as_object().unwrap().clone()),
        )
        .await
        .unwrap();
    assert_eq!(response.is_error, Some(false));
    let value = structured(&response);
    assert_eq!(value["status"], "verified_bundle_consistency");
    assert_eq!(
        value["canonical_bundle_sha256"],
        expected["canonical_sha256"]
    );
    assert_eq!(value["record_count"], expected["record_count"]);
    assert_eq!(
        value["unknown_revision_count"],
        expected["unknown_revision_count"]
    );
    assert!(!value.to_string().contains("local_path"));
    assert!(!value.to_string().contains("normalized_result"));

    client.cancel().await.unwrap();
    server_task.await.unwrap();
}

#[tokio::test]
async fn malformed_bundle_returns_stable_error_category() {
    let (client, server_task) = start_server().await;
    let arguments = json!({ "bundle_json": "{}" });
    let response = client
        .call_tool(
            CallToolRequestParams::new("rea_verify_evidence_bundle_json")
                .with_arguments(arguments.as_object().unwrap().clone()),
        )
        .await
        .unwrap();
    assert_eq!(response.is_error, Some(false));
    assert_eq!(structured(&response)["status"], "rejected");
    assert_eq!(structured(&response)["error_code"], "schema_violation");

    client.cancel().await.unwrap();
    server_task.await.unwrap();
}
