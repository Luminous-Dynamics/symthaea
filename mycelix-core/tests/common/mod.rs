// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use futures_util::StreamExt;
use mycelix_core::RustCoordinator;
use serde_json::Value;
use std::net::TcpListener;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout, Duration};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{error::ProtocolError, protocol::Message, Error as WsError},
};

pub type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

pub fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind free port")
        .local_addr()
        .unwrap()
        .port()
}

pub async fn spawn_coordinator(port: u16, token: &str, max_bytes: usize) -> JoinHandle<()> {
    let coordinator = Arc::new(
        RustCoordinator::new(port)
            .with_auth_token(token.to_string())
            .with_max_message_bytes(max_bytes),
    );
    let (ready_tx, ready_rx) = oneshot::channel();
    let handle = {
        let coordinator = Arc::clone(&coordinator);
        tokio::spawn(async move {
            coordinator
                .start_with_ready(ready_tx)
                .await
                .expect("coordinator start");
        })
    };
    ready_rx.await.expect("coordinator ready");
    handle
}

pub async fn connect_with_retry(url: &str, attempts: usize) -> Result<WsStream, WsError> {
    let mut last_error = None;
    for _ in 0..attempts {
        match connect_async(url).await {
            Ok((stream, _)) => return Ok(stream),
            Err(WsError::Protocol(ProtocolError::HandshakeIncomplete)) => {
                last_error = Some(WsError::Protocol(ProtocolError::HandshakeIncomplete));
                sleep(Duration::from_millis(100)).await;
            }
            Err(err) => return Err(err),
        }
    }
    Err(last_error.unwrap_or(WsError::Protocol(ProtocolError::HandshakeIncomplete)))
}

pub async fn recv_json(ws: &mut WsStream, label: &str) -> Value {
    match timeout(Duration::from_secs(1), ws.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => serde_json::from_str(&text).expect(label),
        Ok(Some(Ok(other))) => panic!("unexpected frame: {:?}", other),
        Ok(Some(Err(e))) => panic!("websocket error: {}", e),
        Ok(None) => panic!("stream closed waiting for {label}"),
        Err(_) => panic!("timeout receiving {label}"),
    }
}

pub struct RegistryGuard {
    _dir: TempDir,
    prev: Option<String>,
}

impl RegistryGuard {
    pub fn new() -> Self {
        let prev = std::env::var("MYCELIX_REGISTRY_PATH").ok();
        let dir = tempfile::tempdir().expect("temp registry dir");
        let path = dir.path().join("agents.json");
        std::env::set_var("MYCELIX_REGISTRY_PATH", &path);
        Self { _dir: dir, prev }
    }
}

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            std::env::set_var("MYCELIX_REGISTRY_PATH", prev);
        } else {
            std::env::remove_var("MYCELIX_REGISTRY_PATH");
        }
    }
}
