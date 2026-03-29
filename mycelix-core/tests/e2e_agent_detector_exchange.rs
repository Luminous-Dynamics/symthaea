// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#[path = "common/mod.rs"]
mod common;
use common::*;

use futures_util::{SinkExt, StreamExt};
use mycelix_core::{ModelUpdate, MycelixAgent};
use serde_json::{json, Value};
use tokio::time::{timeout, Duration};
use tokio_tungstenite::tungstenite::protocol::Message;

#[tokio::test]
async fn e2e_agent_detector_exchange() {
    let _registry = RegistryGuard::new();
    let port = find_free_port();
    let token = "secret-token";
    let server = spawn_coordinator(port, token, 128 * 1024).await;

    let url = format!("ws://127.0.0.1:{port}/?token={token}");
    let mut ws = connect_with_retry(&url, 20)
        .await
        .expect("connect coordinator");

    // Spawn agent
    ws.send(Message::Text(
        json!({"type": "spawn_agent", "agent_id": "alice"}).to_string(),
    ))
    .await
    .unwrap();
    let spawn_resp = recv_json(&mut ws, "spawn response").await;
    assert_eq!(spawn_resp["success"], Value::Bool(true));
    assert_eq!(
        spawn_resp["schema_version"].as_u64(),
        Some(ModelUpdate::SCHEMA_VERSION as u64)
    );
    assert!(
        spawn_resp["agent_info"]["public_key"].as_str().is_some(),
        "spawn response missing public key"
    );

    // Participate in round and capture update
    ws.send(Message::Text(
        json!({"type": "participate_round", "agent_id": "alice", "round_id": 1}).to_string(),
    ))
    .await
    .unwrap();

    let mut update_value: Option<Value> = None;
    for _ in 0..10 {
        if let Ok(Some(Ok(Message::Text(text)))) =
            timeout(Duration::from_millis(250), ws.next()).await
        {
            let value: Value = serde_json::from_str(&text).expect("participate json");
            if value.get("update").is_some() {
                update_value = value.get("update").cloned();
                break;
            }
        }
    }
    let update_json = update_value.expect("coordinator did not return update");
    let update: ModelUpdate = serde_json::from_value(update_json.clone()).expect("model update");
    assert_eq!(update.schema_version, ModelUpdate::SCHEMA_VERSION);

    // Ask coordinator to validate via detector path
    ws.send(Message::Text(
        json!({"type": "validate_update", "update": update_json}).to_string(),
    ))
    .await
    .unwrap();

    let validate_resp = recv_json(&mut ws, "validate response").await;
    assert_eq!(validate_resp["success"], Value::Bool(true));
    assert_eq!(validate_resp["valid"], Value::Bool(true));

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn validate_rejects_unknown_agent() {
    let _registry = RegistryGuard::new();
    let port = find_free_port();
    let token = "secret-token";
    let server = spawn_coordinator(port, token, 128 * 1024).await;

    let url = format!("ws://127.0.0.1:{port}/?token={token}");
    let mut ws = connect_with_retry(&url, 20)
        .await
        .expect("connect coordinator");

    let rogue_update = ModelUpdate {
        schema_version: ModelUpdate::SCHEMA_VERSION,
        agent_id: "ghost".into(),
        round_id: 42,
        weights: vec![0.0; 4],
        timestamp: 0.0,
        signature: String::new(),
        public_key: "unused".into(),
    };

    ws.send(Message::Text(
        json!({"type": "validate_update", "update": rogue_update}).to_string(),
    ))
    .await
    .unwrap();

    let resp = recv_json(&mut ws, "validate response").await;
    assert_eq!(resp["success"], Value::Bool(false));
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or_default()
            .contains("not registered"),
        "unexpected error payload: {:?}",
        resp
    );

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn validation_survives_restart_via_registry() {
    let _registry = RegistryGuard::new();
    let port = find_free_port();
    let token = "secret-token";

    // Boot coordinator and register agent via spawn call.
    let server1 = spawn_coordinator(port, token, 128 * 1024).await;
    let url = format!("ws://127.0.0.1:{port}/?token={token}");
    let mut ws1 = connect_with_retry(&url, 20)
        .await
        .expect("connect coordinator");
    ws1.send(Message::Text(
        json!({"type": "spawn_agent", "agent_id": "alice"}).to_string(),
    ))
    .await
    .unwrap();
    let spawn_resp = recv_json(&mut ws1, "spawn response").await;
    assert_eq!(spawn_resp["success"], Value::Bool(true));
    ws1.close(None).await.ok();
    server1.abort();

    // Produce a deterministic update offline (simulating an agent continuing to operate).
    let mut agent = MycelixAgent::new("alice".into());
    let update = agent.participate_in_round(7).await;
    let update_json = serde_json::to_value(&update).expect("update json");

    // Restart coordinator (reading the existing registry file).
    let server2 = spawn_coordinator(port, token, 128 * 1024).await;
    let mut ws2 = connect_with_retry(&url, 20)
        .await
        .expect("connect coordinator");
    ws2.send(Message::Text(
        json!({"type": "validate_update", "update": update_json}).to_string(),
    ))
    .await
    .unwrap();
    let resp = recv_json(&mut ws2, "validate response").await;
    assert_eq!(resp["success"], Value::Bool(true));
    assert_eq!(resp["valid"], Value::Bool(true));

    ws2.close(None).await.ok();
    server2.abort();
}
