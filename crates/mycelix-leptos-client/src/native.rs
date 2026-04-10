// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native WebSocket transport for testing against a real Holochain conductor.
//!
//! Same wire protocol as `BrowserWsTransport` but uses `tokio-tungstenite`
//! instead of `web_sys::WebSocket`. This allows testing from native Rust
//! without a browser.
//!
//! # Usage
//!
//! ```rust,ignore
//! let transport = NativeWsTransport::new();
//! transport.connect(ConnectConfig {
//!     url: "ws://localhost:8888".into(),
//!     app_id: "mycelix-commons".into(),
//!     auth_token: None,
//! }).await?;
//!
//! let result = transport.call_zome("commons", "proposals", "list", encode(&())?).await?;
//! ```

use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};
use tokio_tungstenite::tungstenite::Message;

use crate::error::ClientError;
use crate::transport::HolochainTransport;
use crate::types::*;

type CellId = (Vec<u8>, Vec<u8>);
type PendingMap = HashMap<u64, oneshot::Sender<Result<Vec<u8>, ClientError>>>;

struct Inner {
    status: ConnectionStatus,
    next_id: u64,
    cell_map: HashMap<String, CellId>,
    agent_pub_key: Option<Vec<u8>>,
    pending: PendingMap,
}

/// Native WebSocket transport for testing.
#[derive(Clone)]
pub struct NativeWsTransport {
    inner: Arc<Mutex<Inner>>,
    write_tx: Arc<
        Mutex<
            Option<
                futures_util::stream::SplitSink<
                    tokio_tungstenite::WebSocketStream<
                        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
                    >,
                    Message,
                >,
            >,
        >,
    >,
}

impl NativeWsTransport {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                status: ConnectionStatus::Disconnected,
                next_id: 1,
                cell_map: HashMap::new(),
                agent_pub_key: None,
                pending: HashMap::new(),
            })),
            write_tx: Arc::new(Mutex::new(None)),
        }
    }

    async fn send_request(
        &self,
        request_type: &str,
        data: Vec<u8>,
    ) -> Result<Vec<u8>, ClientError> {
        let (tx, rx) = oneshot::channel();

        let id = {
            let mut inner = self.inner.lock().await;
            let id = inner.next_id;
            inner.next_id += 1;
            inner.pending.insert(id, tx);
            id
        };

        let wire = WireRequest {
            id,
            request_type: request_type.to_string(),
            data,
        };
        let bytes = rmp_serde::to_vec_named(&wire)
            .map_err(|e| ClientError::SerializationError(e.to_string()))?;

        let mut write_guard = self.write_tx.lock().await;
        let write = write_guard.as_mut().ok_or(ClientError::NotConnected)?;
        write
            .send(Message::Binary(bytes.into()))
            .await
            .map_err(|e| ClientError::WebSocketError(e.to_string()))?;
        drop(write_guard);

        rx.await
            .map_err(|_| ClientError::WebSocketError("channel closed".into()))?
    }
}

impl HolochainTransport for NativeWsTransport {
    fn call_zome(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        payload: Vec<u8>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, ClientError>>>> {
        let role = role_name.to_string();
        let zome = zome_name.to_string();
        let fname = fn_name.to_string();
        let this = self.clone();

        Box::pin(async move {
            let inner = this.inner.lock().await;
            let (dna_hash, agent) = inner
                .cell_map
                .get(&role)
                .ok_or_else(|| ClientError::UnknownRole(role.clone()))?
                .clone();
            let provenance = inner.agent_pub_key.clone().unwrap_or_else(|| agent.clone());
            drop(inner);

            let mut nonce = vec![0u8; 32];
            for (i, b) in nonce.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(37).wrapping_add(42);
            }

            let now_micros = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;

            let req = AppRequest::CallZome(CallZomeRequestWire {
                cell_id: (dna_hash, agent),
                zome_name: zome,
                fn_name: fname.clone(),
                payload,
                cap_secret: None,
                provenance,
                signature: vec![0u8; 64],
                nonce,
                expires_at: now_micros + 5_000_000,
            });

            let data = rmp_serde::to_vec_named(&req)
                .map_err(|e| ClientError::SerializationError(e.to_string()))?;

            this.send_request("request", data).await
        })
    }

    fn status(&self) -> ConnectionStatus {
        // Can't await in a sync fn — return last known status
        ConnectionStatus::Disconnected // Will be overridden after connect
    }

    fn connect(
        &self,
        config: ConnectConfig,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ClientError>>>> {
        let this = self.clone();
        Box::pin(async move {
            // Use cloned `this` instead of `self` to satisfy 'static lifetime
            eprintln!("[NativeWs] Connecting to {}...", config.url);

            // Holochain conductor requires Origin header in WebSocket upgrade
            use tokio_tungstenite::tungstenite::client::IntoClientRequest;
            let mut request = config
                .url
                .clone()
                .into_client_request()
                .map_err(|e| ClientError::ConnectionFailed(e.to_string()))?;
            request
                .headers_mut()
                .insert("Origin", "http://localhost".parse().unwrap());

            let (ws_stream, _) = tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| ClientError::ConnectionFailed(e.to_string()))?;

            let (write, mut read) = ws_stream.split();
            *this.write_tx.lock().await = Some(write);

            // Spawn reader task
            let inner = this.inner.clone();
            tokio::spawn(async move {
                while let Some(msg) = read.next().await {
                    if let Ok(Message::Binary(data)) = msg {
                        if let Ok(resp) = rmp_serde::from_slice::<WireResponse>(&data) {
                            let mut state = inner.lock().await;
                            if let Some(tx) = state.pending.remove(&resp.id) {
                                if let Some(err) = resp.error {
                                    let _ = tx.send(Err(ClientError::ZomeCallFailed(err)));
                                } else {
                                    let _ = tx.send(Ok(resp.data));
                                }
                            }
                        }
                    }
                }
            });

            // Authenticate if token provided
            if let Some(token) = config.auth_token {
                let auth_req = AppRequest::Authenticate { token };
                let data = rmp_serde::to_vec_named(&auth_req)
                    .map_err(|e| ClientError::SerializationError(e.to_string()))?;
                let _ = this.send_request("authenticate", data).await;
            }

            // App Info discovery
            let info_req = AppRequest::AppInfo {
                installed_app_id: config.app_id.clone(),
            };
            let data = rmp_serde::to_vec_named(&info_req)
                .map_err(|e| ClientError::SerializationError(e.to_string()))?;

            let info_bytes = this.send_request("request", data).await?;

            let info: AppInfoResponse = rmp_serde::from_slice(&info_bytes)
                .map_err(|e| ClientError::InvalidResponse(format!("app_info decode: {e}")))?;

            eprintln!(
                "[NativeWs] App: {}, {} roles",
                info.installed_app_id,
                info.cell_info.len()
            );

            let mut inner = this.inner.lock().await;
            for entry in &info.cell_info {
                for cell in &entry.cells {
                    if let CellInfoVariant::Provisioned(p) = cell {
                        inner
                            .cell_map
                            .insert(entry.role_name.clone(), p.cell_id.clone());
                        if inner.agent_pub_key.is_none() {
                            inner.agent_pub_key = Some(p.cell_id.1.clone());
                        }
                        eprintln!("[NativeWs] Role '{}' -> cell discovered", entry.role_name);
                    }
                }
            }
            inner.status = ConnectionStatus::Connected;

            eprintln!(
                "[NativeWs] Connected! {} roles mapped",
                inner.cell_map.len()
            );
            Ok(())
        })
    }

    fn disconnect(&self) {
        // Drop the write half — reader will close naturally
        let write_tx = self.write_tx.clone();
        tokio::spawn(async move {
            *write_tx.lock().await = None;
        });
    }
}
