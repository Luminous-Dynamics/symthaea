// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conductor client — relays IMAP messages into Holochain DHT.

use anyhow::Result;

#[derive(Clone)]
pub struct ConductorClient {
    url: String,
}

impl ConductorClient {
    pub async fn connect(url: &str) -> Result<Self> {
        // Verify connectivity with a simple TCP check
        let addr = url.trim_start_matches("ws://").trim_start_matches("wss://");
        let _ = tokio::net::TcpStream::connect(addr).await
            .map_err(|e| anyhow::anyhow!("Cannot reach conductor at {url}: {e}"))?;
        tracing::info!("Conductor connectivity verified at {url}");
        Ok(Self { url: url.to_string() })
    }

    /// Relay an inbound IMAP message into Holochain via federation bridge.
    pub async fn relay_inbound(
        &self,
        from: &str,
        subject: &str,
        body: &str,
        receiving_account: &str,
    ) -> Result<()> {
        use futures_util::SinkExt;
        use tokio_tungstenite::tungstenite::Message;

        let (mut ws, _) = tokio_tungstenite::connect_async(&self.url).await?;

        let envelope = serde_json::json!({
            "type": "call_zome",
            "data": {
                "zome_name": "mail_federation",
                "fn_name": "receive_federated_envelope",
                "payload": {
                    "envelope_id": format!("imap-{}-{}", receiving_account, chrono::Utc::now().timestamp()),
                    "source_network": "smtp",
                    "source_agent": from,
                    "dest_network": "holochain-native",
                    "dest_agent": receiving_account,
                    "encrypted_payload": body.as_bytes().to_vec(),
                    "timestamp": chrono::Utc::now().timestamp(),
                    "ttl": 86400,
                    "hop_count": 0,
                    "max_hops": 1,
                    "previous_hops": [],
                    "priority": "normal",
                    "status": "delivered",
                    "metadata": {
                        "subject": subject,
                        "from": from,
                        "protocol": "imap",
                    }
                }
            }
        });

        let payload = rmp_serde::to_vec_named(&envelope)?;
        ws.send(Message::Binary(payload.into())).await?;

        tracing::debug!("Relayed inbound from {} to conductor", from);
        Ok(())
    }
}
