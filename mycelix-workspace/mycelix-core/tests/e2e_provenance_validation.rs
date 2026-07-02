#![allow(clippy::module_inception)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[path = "common/mod.rs"]
mod common;

use blake3::Hasher;
use common::*;
use futures_util::SinkExt;
use httpmock::Method::POST;
use httpmock::MockServer;
use mycelix_core::{AgentRegistry, LossDeltaInput, ModelUpdate, MycelixAgent, ProofService};
use serde_json::{json, Value};
use tokio::time::{sleep, Duration};
use tokio_tungstenite::tungstenite::protocol::Message;

#[tokio::test]
async fn provenance_survives_restart_and_is_verifiable() {
    let ipfs_server = MockServer::start_async().await;
    let _mock = ipfs_server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/api/v0/add")
                .query_param("pin", "false");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"Name":"payload","Hash":"QmTestCID"}"#);
        })
        .await;

    let prev_ipfs = std::env::var("MYCELIX_IPFS_API").ok();
    std::env::set_var("MYCELIX_IPFS_API", ipfs_server.base_url());

    let _registry_guard = RegistryGuard::new();
    let port = find_free_port();
    let token = "provenance-token";
    let url = format!("ws://127.0.0.1:{port}/?token={token}");

    // boot coordinator and spawn agent
    let server = spawn_coordinator(port, token, 128 * 1024).await;
    let mut ws = connect_with_retry(&url, 20).await.expect("connect");
    ws.send(Message::Text(
        json!({"type": "spawn_agent", "agent_id": "alice"}).to_string(),
    ))
    .await
    .unwrap();
    recv_json(&mut ws, "spawn response").await;
    ws.close(None).await.ok();

    // participate and validate update
    let mut agent = MycelixAgent::new("alice".into());
    let update = agent.participate_in_round(9).await;
    let update_json = serde_json::to_value(&update).unwrap();

    let mut ws = connect_with_retry(&url, 20).await.expect("connect");
    ws.send(Message::Text(
        json!({"type": "validate_update", "update": update_json}).to_string(),
    ))
    .await
    .unwrap();
    let resp = recv_json(&mut ws, "validate response").await;
    assert_eq!(resp["success"], Value::Bool(true));
    assert_eq!(resp["valid"], Value::Bool(true));
    ws.close(None).await.ok();
    sleep(Duration::from_millis(200)).await;
    _mock.assert_async().await;
    server.abort();

    // verify registry archive entry
    let registry_path = std::env::var("MYCELIX_REGISTRY_PATH").unwrap();
    let registry = AgentRegistry::new(registry_path);
    let record = registry.get("alice").await.expect("registry record");
    assert_eq!(record.archives.len(), 1);
    let entry = &record.archives[0];
    assert_eq!(entry.round_id, 9);
    assert_eq!(entry.cid, "QmTestCID");

    // inspect mock IPFS payload and re-verify proof
    // Recompute proof deterministically and verify
    let input = loss_input_from_update(&update);
    let proof = ProofService::prove_loss_delta(&input).expect("prove");
    let output = ProofService::verify_loss_delta(&proof).expect("verify");
    assert!(output.passed);

    if let Some(prev) = prev_ipfs {
        std::env::set_var("MYCELIX_IPFS_API", prev);
    } else {
        std::env::remove_var("MYCELIX_IPFS_API");
    }
}

fn loss_input_from_update(update: &ModelUpdate) -> LossDeltaInput {
    let mut hasher = Hasher::new();
    for weight in &update.weights {
        hasher.update(&weight.to_le_bytes());
    }
    LossDeltaInput {
        round_id: update.round_id,
        model_hash: hasher.finalize().to_hex().to_string(),
        baseline_loss: update.weights.first().copied().unwrap_or(0.5).abs(),
        new_loss: update.weights.last().copied().unwrap_or(0.4).abs(),
        tolerance: 1.0,
    }
}
