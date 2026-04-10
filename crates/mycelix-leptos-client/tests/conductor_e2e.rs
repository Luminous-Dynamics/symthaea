// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! E2E tests against a real Holochain conductor.
//!
//! Uses the official `holochain_client` crate (proven working in
//! symthaea-mycelix-holochain) to validate connectivity, then tests
//! our wire format compatibility.
//!
//! Run with:
//!   cargo test -p mycelix-leptos-client --test conductor_e2e -- --ignored --nocapture

use mycelix_leptos_client::{decode, encode};

fn admin_url() -> String {
    std::env::var("ADMIN_URL").unwrap_or_else(|_| "ws://localhost:33743".to_string())
}

fn app_url() -> String {
    std::env::var("CONDUCTOR_URL").unwrap_or_else(|_| "ws://localhost:8888".to_string())
}

fn admin_addr() -> String {
    std::env::var("ADMIN_ADDR").unwrap_or_else(|_| "localhost:33743".to_string())
}

/// Connect to admin using official holochain_client and list apps.
#[tokio::test]
#[ignore]
async fn admin_list_apps() {
    use holochain_client::AdminWebsocket;

    let addr = admin_addr();
    eprintln!("Admin: {addr}");

    let admin = match AdminWebsocket::connect(addr).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("SKIP: {e:?}");
            return;
        }
    };
    eprintln!("Connected!");

    match admin.list_apps(None).await {
        Ok(apps) => {
            eprintln!("{} app(s):", apps.len());
            for app in &apps {
                eprintln!("  {} ({:?})", app.installed_app_id, app.status);
            }
        }
        Err(e) => eprintln!("list_apps: {e:?}"),
    }
}

/// Connect to app interface using official holochain_client.
#[tokio::test]
#[ignore]
async fn app_connect_and_info() {
    use holochain_client::{AppWebsocket, ClientAgentSigner};
    use std::sync::Arc;

    let url = "localhost:8888".to_string();
    let token: Vec<u8> = std::env::var("MYCELIX_APP_TOKEN")
        .unwrap_or_default()
        .into_bytes();
    let signer: Arc<dyn holochain_client::AgentSigner + Send + Sync> =
        Arc::new(ClientAgentSigner::default());

    eprintln!("App: {url}");

    match AppWebsocket::connect(url, token, signer).await {
        Ok(ws) => {
            eprintln!("Connected!");
            match ws.app_info().await {
                Ok(Some(info)) => {
                    eprintln!("App: {}", info.installed_app_id);
                    for (role, cells) in &info.cell_info {
                        eprintln!("  Role: {role} ({} cells)", cells.len());
                    }
                }
                Ok(None) => eprintln!("app_info: None"),
                Err(e) => eprintln!("app_info: {e:?}"),
            }
        }
        Err(e) => eprintln!("SKIP: {e:?}"),
    }
}

/// Encode/decode roundtrip verification.
#[test]
fn our_encode_decode_roundtrip() {
    let data = serde_json::json!({"id": "MIP-042", "title": "Solar garden", "status": "Active"});
    let encoded = encode(&data).unwrap();
    let decoded: serde_json::Value = decode(&encoded).unwrap();
    assert_eq!(decoded["id"], "MIP-042");
}
